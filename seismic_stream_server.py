#!/usr/bin/env python3
"""
Seismic Data Stream Server
==========================

Builds a long, realistic seismic waveform by stitching together:
  - Background noise segments  (from extracted_waves/noise/*.npy)
  - Real earthquake recordings (from dataset/pyweed/HXZ/*.mseed)

The composite waveform alternates between noise and earthquake events,
simulating what a live seismometer would capture over hours.

The server streams this data over TCP so the SeismicQuake desktop
application can consume it as a live real-time feed.

Protocol (per frame):
    4 bytes  – int32 little-endian: number of samples N in this frame
    N×4 bytes – float32 little-endian: sample values

Usage:
    # Build the composite waveform and start streaming on port 9100
    python seismic_stream_server.py --port 9100

    # Build with custom settings
    python seismic_stream_server.py --port 9100 --duration 3600 --eq-interval 120

    # Build-only mode (save to .npy without serving)
    python seismic_stream_server.py --build-only --output stream_data.npy
"""

import os
import sys
import time
import glob
import struct
import socket
import signal
import random
import argparse
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 100          # Hz – must match SeismicQuake models
FRAME_SAMPLES = 100        # samples per TCP frame (1 second at 100 Hz)
FRAME_INTERVAL = 1.0       # seconds between frames (real-time pacing)

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent
NOISE_DIR = PROJECT_ROOT / "extracted_waves" / "noise"
P_WAVE_DIR = PROJECT_ROOT / "extracted_waves" / "p_wave"
S_WAVE_DIR = PROJECT_ROOT / "extracted_waves" / "s_wave"
SURFACE_DIR = PROJECT_ROOT / "extracted_waves" / "surface_wave"
PYWEED_DIR = PROJECT_ROOT / "dataset" / "pyweed"


# ---------------------------------------------------------------------------
# Waveform Builder
# ---------------------------------------------------------------------------
class SeismicStreamBuilder:
    """
    Constructs a long continuous seismic waveform by interleaving
    noise intervals with real earthquake recordings.
    """

    def __init__(self, target_duration: float = 1800,
                 eq_interval_range: tuple = (60, 180),
                 noise_amplitude: float = 0.05,
                 verbose: bool = True):
        """
        Args:
            target_duration : Total stream duration in seconds.
            eq_interval_range : (min, max) seconds between earthquake events.
            noise_amplitude : Scale factor for background noise level.
            verbose : Print build progress.
        """
        self.target_duration = target_duration
        self.eq_interval_min, self.eq_interval_max = eq_interval_range
        self.noise_amplitude = noise_amplitude
        self.verbose = verbose
        self.sample_rate = SAMPLE_RATE

        # Loaded waveform pools
        self._noise_pool: list = []
        self._earthquake_pool: list = []

    # ----- data loading -----

    def _log(self, msg):
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {msg}")

    def _load_noise_pool(self, max_files: int = 500):
        """Load a random subset of noise .npy files."""
        noise_files = sorted(glob.glob(str(NOISE_DIR / "*.npy")))
        if not noise_files:
            self._log("⚠  No noise files found – will generate synthetic noise.")
            return

        subset = random.sample(noise_files, min(max_files, len(noise_files)))
        for f in subset:
            try:
                arr = np.load(f).astype(np.float32)
                # Normalize to [-1, 1]
                mx = np.max(np.abs(arr))
                if mx > 0:
                    arr = arr / mx
                self._noise_pool.append(arr)
            except Exception:
                pass

        self._log(f"✓ Loaded {len(self._noise_pool)} noise segments "
                  f"({self._noise_pool[0].shape[0]} samples each)")

    def _load_earthquake_pool(self, max_events: int = 60):
        """
        Build earthquake events from extracted_waves .npy files.
        
        The AI models were trained on individual wave phases (each as a
        standalone segment of up to 400 samples). We build each earthquake
        event as a sequence of separate phases padded to 400 samples each,
        separated by short noise gaps. This ensures the sliding detection
        window always sees a waveform shape matching the training data.
        
        Event structure:
            [P-wave padded to 400] -> [noise gap] -> [S-wave padded to 400]
            -> [noise gap] -> [Surface wave padded to 400]
        """
        p_files = sorted(glob.glob(str(P_WAVE_DIR / "*.npy")))
        s_files = sorted(glob.glob(str(S_WAVE_DIR / "*.npy")))
        sf_files = sorted(glob.glob(str(SURFACE_DIR / "*.npy")))

        if not (p_files and s_files and sf_files):
            self._log("Warning: No extracted wave files found!")
            return

        n_events = min(max_events, len(p_files), len(s_files), len(sf_files))
        indices = random.sample(
            range(min(len(p_files), len(s_files), len(sf_files))),
            n_events
        )

        input_len = 400  # Must match INPUT_LENGTH in seismic_analyzer

        for idx in indices:
            try:
                p_raw = np.load(p_files[idx]).astype(np.float32)
                s_raw = np.load(s_files[idx]).astype(np.float32)
                sf_raw = np.load(sf_files[idx]).astype(np.float32)

                # Pad each phase to exactly INPUT_LENGTH (400 samples)
                # This matches the training data format
                def pad_to_length(arr, length):
                    if len(arr) >= length:
                        return arr[:length]
                    padded = np.zeros(length, dtype=np.float32)
                    padded[:len(arr)] = arr
                    return padded

                p = pad_to_length(p_raw, input_len)
                s = pad_to_length(s_raw, input_len)
                sf = pad_to_length(sf_raw, input_len)

                # Create short noise gaps between phases (0.5-2s)
                gap1_samples = int(SAMPLE_RATE * random.uniform(0.5, 2.0))
                gap2_samples = int(SAMPLE_RATE * random.uniform(1.0, 3.0))
                gap1 = np.random.randn(gap1_samples).astype(np.float32) * 0.01
                gap2 = np.random.randn(gap2_samples).astype(np.float32) * 0.01

                # Build event: P(400) -> gap -> S(400) -> gap -> Surface(400)
                composite = np.concatenate([p, gap1, s, gap2, sf])
                self._earthquake_pool.append(composite)
            except Exception:
                pass

        self._log(f"Built {len(self._earthquake_pool)} earthquake events "
                  f"from extracted P/S/Surface waves (each phase padded to {input_len} samples)")

        if not self._earthquake_pool:
            self._log("Warning: No earthquake data loaded! Stream will be noise-only.")

    # ----- waveform construction -----

    def _generate_noise_segment(self, duration_seconds: float) -> np.ndarray:
        """Generate a noise segment of given duration."""
        n_samples = int(duration_seconds * SAMPLE_RATE)

        if self._noise_pool:
            # Tile noise snippets to fill the duration
            segments = []
            total = 0
            while total < n_samples:
                snip = random.choice(self._noise_pool).copy()
                snip *= self.noise_amplitude
                # Add slight random variation
                snip += np.random.randn(len(snip)).astype(np.float32) * 0.005
                segments.append(snip)
                total += len(snip)
            noise = np.concatenate(segments)[:n_samples]
        else:
            noise = np.random.randn(n_samples).astype(np.float32) * self.noise_amplitude

        return noise

    def _insert_earthquake(self, eq_waveform: np.ndarray) -> np.ndarray:
        """
        Prepare an earthquake waveform for insertion.
        Applies fade-in/out to blend smoothly with surrounding noise.
        Raw amplitudes are preserved since the AI normalizes per-window.
        """
        eq = eq_waveform.copy()

        # Smooth fade-in/out to avoid sharp discontinuities
        fade_len = min(50, len(eq) // 4)
        if fade_len > 0:
            eq[:fade_len] *= np.linspace(0, 1, fade_len)
            eq[-fade_len:] *= np.linspace(1, 0, fade_len)

        return eq

    def build(self) -> np.ndarray:
        """
        Build the complete composite waveform.

        Returns:
            np.ndarray of float32 samples at SAMPLE_RATE Hz
        """
        self._log(f"Building {self.target_duration}s composite stream at {SAMPLE_RATE} Hz...")
        self._log("")

        # Load data pools
        self._load_noise_pool()
        self._load_earthquake_pool()

        self._log("")

        total_samples = int(self.target_duration * SAMPLE_RATE)
        stream = np.zeros(total_samples, dtype=np.float32)
        cursor = 0
        event_log = []

        # Start with an initial noise period
        initial_noise_dur = random.uniform(15, 45)
        noise = self._generate_noise_segment(initial_noise_dur)
        n = min(len(noise), total_samples - cursor)
        stream[cursor:cursor + n] = noise[:n]
        cursor += n

        event_num = 0

        while cursor < total_samples:
            # Insert earthquake event
            if self._earthquake_pool:
                eq = random.choice(self._earthquake_pool)
                eq = self._insert_earthquake(eq)

                n = min(len(eq), total_samples - cursor)
                # Blend: add earthquake on top of low-level noise
                background = np.random.randn(n).astype(np.float32) * self.noise_amplitude * 0.3
                stream[cursor:cursor + n] = eq[:n] + background

                event_num += 1
                event_time = cursor / SAMPLE_RATE
                event_dur = n / SAMPLE_RATE
                event_log.append({
                    "event": event_num,
                    "start_time": round(event_time, 2),
                    "duration": round(event_dur, 2),
                    "start_sample": cursor,
                    "end_sample": cursor + n
                })
                self._log(f"  🌊 Event #{event_num} at {event_time:.1f}s "
                          f"(duration: {event_dur:.1f}s)")

                cursor += n

            # Insert noise gap until next event
            gap_duration = random.uniform(self.eq_interval_min, self.eq_interval_max)
            noise = self._generate_noise_segment(gap_duration)
            n = min(len(noise), total_samples - cursor)
            stream[cursor:cursor + n] = noise[:n]
            cursor += n

        # IMPORTANT: Keep raw float32 amplitudes without clipping or normalizing.
        # The analyzer's _preprocess_segment() normalizes each 400-sample window
        # individually, so the waveform SHAPE is what matters for detection.
        # Clipping would destroy the waveform shape of earthquake signals.

        self._log("")
        self._log(f"✓ Stream built: {len(stream)} samples, "
                  f"{len(stream)/SAMPLE_RATE:.1f}s, "
                  f"{event_num} earthquake events embedded")
        self._log(f"  Event timeline:")
        for e in event_log:
            self._log(f"    Event #{e['event']}: {e['start_time']}s – "
                      f"{e['start_time']+e['duration']:.1f}s")

        return stream


# ---------------------------------------------------------------------------
# TCP Streaming Server
# ---------------------------------------------------------------------------
class SeismicStreamServer:
    """
    TCP server that streams seismic data to connected clients.

    Protocol per frame:
        [4 bytes: int32 LE sample count N]
        [N * 4 bytes: float32 LE samples]

    The server loops the waveform data continuously.
    """

    def __init__(self, data: np.ndarray, host: str = "0.0.0.0",
                 port: int = 9100, speed: float = 1.0):
        self.data = data.astype(np.float32)
        self.host = host
        self.port = port
        self.speed = max(0.1, speed)
        self.running = False
        self.clients: list = []
        self.lock = threading.Lock()
        self._server_socket = None

    def _accept_clients(self):
        """Accept incoming client connections."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)

        print(f"\n{'='*60}")
        print(f"🌐 Seismic Stream Server")
        print(f"   Listening on {self.host}:{self.port}")
        print(f"   Stream: {len(self.data)} samples "
              f"({len(self.data)/SAMPLE_RATE:.1f}s at {SAMPLE_RATE} Hz)")
        print(f"   Speed: {self.speed:.1f}x real-time")
        print(f"   Frame: {FRAME_SAMPLES} samples / frame")
        print(f"{'='*60}")
        print(f"\nWaiting for clients... (Ctrl+C to stop)\n")

        while self.running:
            try:
                conn, addr = self._server_socket.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.lock:
                    self.clients.append(conn)
                print(f"  ✓ Client connected: {addr[0]}:{addr[1]} "
                      f"(total: {len(self.clients)})")
            except socket.timeout:
                continue
            except OSError:
                break

    def _stream_data(self):
        """Stream data frames to all connected clients."""
        cursor = 0
        total = len(self.data)
        frame_interval = FRAME_INTERVAL / self.speed
        loop_count = 0

        while self.running:
            # Extract frame
            end = cursor + FRAME_SAMPLES
            if end > total:
                # Loop back to beginning
                frame = np.concatenate([
                    self.data[cursor:],
                    self.data[:end - total]
                ])
                cursor = end - total
                loop_count += 1
                print(f"\n  ↻ Stream looped (cycle {loop_count})\n")
            else:
                frame = self.data[cursor:end]
                cursor = end

            # Build TCP frame: [N:int32][samples:float32[N]]
            n = len(frame)
            header = struct.pack('<i', n)
            payload = frame.tobytes()  # float32 LE

            # Send to all clients
            dead_clients = []
            with self.lock:
                for client in self.clients:
                    try:
                        client.sendall(header + payload)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        dead_clients.append(client)

                # Remove disconnected clients
                for dc in dead_clients:
                    self.clients.remove(dc)
                    try:
                        dc.close()
                    except Exception:
                        pass
                    print(f"  ✗ Client disconnected (remaining: {len(self.clients)})")

            # Real-time pacing
            time.sleep(frame_interval)

    def start(self):
        """Start the streaming server."""
        self.running = True

        # Start acceptor thread
        acceptor = threading.Thread(target=self._accept_clients, daemon=True)
        acceptor.start()

        # Start streamer in main thread
        try:
            self._stream_data()
        except KeyboardInterrupt:
            print("\n\n  Server shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the server and close all connections."""
        self.running = False

        with self.lock:
            for client in self.clients:
                try:
                    client.close()
                except Exception:
                    pass
            self.clients.clear()

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

        print("  ✓ Server stopped.\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Seismic Data Stream Server – builds composite waveforms "
                    "and streams them over TCP for real-time earthquake detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on port 9100 with 30-minute stream
  python seismic_stream_server.py --port 9100 --duration 1800

  # Fast-forward at 5x speed
  python seismic_stream_server.py --port 9100 --speed 5.0

  # Build and save waveform to file without serving
  python seismic_stream_server.py --build-only --output stream.npy

  # Multiple ports (run multiple instances)
  python seismic_stream_server.py --port 9100 &
  python seismic_stream_server.py --port 9101 &
        """
    )

    parser.add_argument("--port", type=int, default=9100,
                        help="TCP port to serve on (default: 9100)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--duration", type=float, default=1800,
                        help="Total stream duration in seconds (default: 1800 = 30 min)")
    parser.add_argument("--eq-interval", type=float, nargs=2, default=[60, 180],
                        metavar=("MIN", "MAX"),
                        help="Min/max seconds between earthquake events (default: 60 180)")
    parser.add_argument("--noise-level", type=float, default=0.05,
                        help="Background noise amplitude (default: 0.05)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Streaming speed multiplier (default: 1.0 = real-time)")
    parser.add_argument("--build-only", action="store_true",
                        help="Build the waveform and save to file, don't start server")
    parser.add_argument("--output", type=str, default="stream_data.npy",
                        help="Output file for --build-only mode (default: stream_data.npy)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible streams")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Build the composite waveform
    builder = SeismicStreamBuilder(
        target_duration=args.duration,
        eq_interval_range=tuple(args.eq_interval),
        noise_amplitude=args.noise_level
    )
    stream_data = builder.build()

    if args.build_only:
        out_path = PROJECT_ROOT / args.output
        np.save(out_path, stream_data)
        print(f"\n✓ Saved stream to: {out_path}")
        print(f"  {len(stream_data)} samples, {len(stream_data)/SAMPLE_RATE:.1f}s")
        return

    # Start TCP server
    server = SeismicStreamServer(
        data=stream_data,
        host=args.host,
        port=args.port,
        speed=args.speed
    )

    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server.start()


if __name__ == "__main__":
    main()
