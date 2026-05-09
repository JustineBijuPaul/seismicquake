#!/usr/bin/env python3
"""
Earthquake Detection Desktop Application

A PyQt6-based desktop application for real-time earthquake detection
and seismic wave classification using trained AI models.

Features:
- Load and analyze .mseed, .wav, and .npy seismic files
- Real-time waveform visualization
- P, S, and Surface wave classification
- Magnitude prediction from P-waves
- Real-time monitoring mode with alerts
- Export results to JSON/CSV

Usage:
    python earthquake_desktop_app.py
"""

import os
import sys
import json
import time
import struct
import socket
import warnings
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Optional, List, Tuple
from collections import deque

# Suppress warnings and TensorFlow noise
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QStatusBar,
    QTabWidget, QGroupBox, QGridLayout, QTextEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QSlider, QSplitter,
    QFrame, QListWidget, QListWidgetItem, QMessageBox, QToolBar,
    QSizePolicy, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QDialogButtonBox, QFormLayout,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QUrl,
    QPropertyAnimation
)
from PyQt6.QtGui import (
    QAction, QIcon, QFont, QPalette, QColor, QPainter, QPen,
    QBrush, QLinearGradient, QPixmap, QDragEnterEvent, QDropEvent
)

# Matplotlib for plotting
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import our seismic analyzer
from seismic_analyzer import (
    SeismicAnalyzer, RealtimeMonitor, WaveDetection, AnalysisResult,
    INPUT_LENGTH, SAMPLE_RATE
)


# =============================================================================
# Style Constants
# =============================================================================
COLORS = {
    'P': '#6366F1',      # Indigo 500
    'S': '#10B981',      # Emerald 500
    'Surface': '#F59E0B', # Amber 500
    'p_wave': '#6366F1',
    's_wave': '#10B981',
    'surface_wave': '#F59E0B',
    'Noise': '#94A3B8',   # Slate 400
    'background': '#0F172A', # Slate 900
    'surface': '#1E293B',    # Slate 800
    'border': '#334155',     # Slate 700
    'primary': '#6366F1',    # Indigo 500
    'primary_hover': '#818CF8', # Indigo 400
    'accent': '#F43F5E',     # Rose 500
    'text': '#F1F5F9',       # Slate 100
    'text_muted': '#94A3B8', # Slate 400
    'success': '#10B981',    # Emerald 500
    'warning': '#F59E0B',    # Amber 500
    'error': '#EF4444'       # Red 500
}

DARK_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}
QWidget {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    margin-top: 20px;
    padding-top: 15px;
    background-color: {COLORS['surface']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 10px;
    color: {COLORS['primary_hover']};
    font-weight: bold;
    font-size: 14px;
}}
QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {COLORS['primary_hover']};
}}
QPushButton:pressed {{
    background-color: {COLORS['accent']};
    padding-top: 11px;
    padding-bottom: 9px;
}}
QPushButton:disabled {{
    background-color: #334155;
    color: #64748b;
}}
QPushButton#startButton {{
    background-color: {COLORS['success']};
}}
QPushButton#startButton:hover {{
    background-color: #34d399;
}}
QPushButton#stopButton {{
    background-color: {COLORS['error']};
}}
QPushButton#stopButton:hover {{
    background-color: #f87171;
}}
QProgressBar {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    text-align: center;
    background-color: {COLORS['background']};
    height: 12px;
}}
QProgressBar::chunk {{
    background-color: {COLORS['primary']};
    border-radius: 5px;
}}
QTextEdit, QListWidget, QTableWidget {{
    background-color: {COLORS['background']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px;
}}
QHeaderView::section {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_muted']};
    padding: 12px;
    border: none;
    border-bottom: 2px solid {COLORS['border']};
    font-weight: bold;
}}
QTabWidget::pane {{
    border: none;
    background-color: {COLORS['background']};
}}
QTabBar::tab {{
    background-color: transparent;
    color: {COLORS['text_muted']};
    padding: 12px 24px;
    margin-right: 4px;
    border-bottom: 3px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:hover {{
    color: {COLORS['text']};
    background-color: {COLORS['surface']};
    border-radius: 8px;
}}
QTabBar::tab:selected {{
    color: {COLORS['primary_hover']};
    border-bottom: 3px solid {COLORS['primary']};
    background-color: {COLORS['surface']};
    border-radius: 8px 8px 0 0;
}}
QComboBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 12px;
}}
QComboBox::drop-down {{
    border: none;
    width: 30px;
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px;
}}
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {COLORS['border']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['primary']};
    border: 2px solid {COLORS['background']};
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{
    background: {COLORS['primary_hover']};
}}
QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {COLORS['border']};
    border-radius: 5px;
    min-height: 40px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['text_muted']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QStatusBar {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_muted']};
    border-top: 1px solid {COLORS['border']};
}}
QToolBar {{
    background-color: {COLORS['surface']};
    border-bottom: 1px solid {COLORS['border']};
    spacing: 12px;
    padding: 8px;
}}
QMenuBar {{
    background-color: {COLORS['surface']};
    border-bottom: 1px solid {COLORS['border']};
}}
QMenuBar::item {{
    padding: 8px 16px;
    background-color: transparent;
}}
QMenuBar::item:selected {{
    background-color: {COLORS['border']};
    border-radius: 6px;
}}
QMenu {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {COLORS['primary']};
    color: white;
}}
"""


# =============================================================================
# Worker Threads
# =============================================================================
class AnalysisWorker(QThread):
    """Worker thread for file analysis."""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    
    def __init__(self, analyzer: SeismicAnalyzer, filepath: str):
        super().__init__()
        self.analyzer = analyzer
        self.filepath = filepath
    
    def run(self):
        try:
            self.progress.emit(10, "Loading file...")
            data, sample_rate = self.analyzer.load_file(self.filepath)
            
            self.progress.emit(30, "Analyzing waveform...")
            result = self.analyzer.analyze_file(self.filepath)
            
            self.progress.emit(100, "Complete!")
            self.finished.emit((result, data, sample_rate))
        except Exception as e:
            self.error.emit(str(e))


class RealtimeWorker(QThread):
    """Worker thread for real-time monitoring."""
    detection = pyqtSignal(object, float)  # WaveDetection, timestamp
    sample_update = pyqtSignal(np.ndarray)  # Latest samples for visualization
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, analyzer: SeismicAnalyzer, data: np.ndarray, sample_rate: float):
        super().__init__()
        self.analyzer = analyzer
        self.data = data
        self.sample_rate = sample_rate
        self.is_running = True
        self.speed = 1.0  # Playback speed multiplier
    
    def stop(self):
        self.is_running = False
    
    def set_speed(self, speed: float):
        self.speed = max(0.1, min(10.0, speed))
    
    def run(self):
        buffer = deque(maxlen=INPUT_LENGTH * 2)
        chunk_size = int(self.sample_rate / 10)  # 100ms chunks
        last_detection_time = 0
        cooldown = 2.0
        
        self.status_update.emit("Monitoring started...")
        
        i = 0
        while i < len(self.data) and self.is_running:
            chunk = self.data[i:i + chunk_size]
            buffer.extend(chunk)
            
            # Emit samples for visualization
            if len(buffer) >= INPUT_LENGTH:
                self.sample_update.emit(np.array(list(buffer)[-INPUT_LENGTH * 2:]))
            
            # Check for earthquake every chunk
            if len(buffer) >= INPUT_LENGTH:
                current_time = i / self.sample_rate
                
                if current_time - last_detection_time >= cooldown:
                    segment = np.array(list(buffer)[-INPUT_LENGTH:], dtype=np.float32)
                    is_eq, confidence = self.analyzer.detect_earthquake(segment)
                    
                    if is_eq:
                        wave_type, wave_conf = self.analyzer.classify_wave(segment)
                        magnitude = None
                        if wave_type == 'P':
                            magnitude, _ = self.analyzer.predict_magnitude(segment)
                        
                        detection = WaveDetection(
                            wave_type=wave_type,
                            confidence=wave_conf,
                            start_sample=i - INPUT_LENGTH,
                            end_sample=i,
                            start_time=current_time - INPUT_LENGTH / self.sample_rate,
                            end_time=current_time,
                            magnitude=magnitude
                        )
                        self.detection.emit(detection, current_time)
                        last_detection_time = current_time
            
            i += chunk_size
            time.sleep((chunk_size / self.sample_rate) / self.speed)
        
        self.status_update.emit("Monitoring complete")
        self.finished.emit()


class TcpStreamWorker(QThread):
    """Worker thread that receives seismic data from a TCP stream server."""
    detection = pyqtSignal(object, float)  # WaveDetection, timestamp
    sample_update = pyqtSignal(np.ndarray)  # Latest samples for visualization
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, analyzer: SeismicAnalyzer, host: str, port: int):
        super().__init__()
        self.analyzer = analyzer
        self.host = host
        self.port = port
        self.is_running = True
        self.sample_rate = SAMPLE_RATE
    
    def stop(self):
        self.is_running = False
    
    def set_speed(self, speed: float):
        """Speed is controlled server-side for TCP streams; this is a no-op."""
        pass
    
    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n and self.is_running:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    raise ConnectionError("Server closed connection")
                data += chunk
            except socket.timeout:
                continue
        return data
    
    def run(self):
        buffer = deque(maxlen=INPUT_LENGTH * 4)
        last_detection_time = 0
        cooldown = 2.0
        total_samples = 0
        
        self.status_update.emit(f"Connecting to {self.host}:{self.port}...")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((self.host, self.port))
            sock.settimeout(2.0)
            self.status_update.emit(f"🟢 Connected to stream at {self.host}:{self.port}")
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            error_msg = f"❌ Connection failed: {e}"
            if "111" in str(e) or "refused" in str(e).lower():
                error_msg += "\n💡 TIP: Is the 'seismic_stream_server.py' running?"
            self.status_update.emit(error_msg)
            self.finished.emit()
            return
        
        try:
            while self.is_running:
                # Read frame header: 4 bytes int32 LE (sample count)
                header = self._recv_exact(sock, 4)
                if len(header) < 4:
                    break
                
                n_samples = struct.unpack('<i', header)[0]
                
                if n_samples <= 0 or n_samples > 100000:
                    continue  # Skip invalid frames
                
                # Read sample data: N * 4 bytes float32 LE
                payload = self._recv_exact(sock, n_samples * 4)
                if len(payload) < n_samples * 4:
                    break
                
                samples = np.frombuffer(payload, dtype=np.float32).copy()
                buffer.extend(samples)
                total_samples += len(samples)
                current_time = total_samples / self.sample_rate
                
                # Emit samples for visualization
                if len(buffer) >= INPUT_LENGTH:
                    vis_data = np.array(list(buffer)[-INPUT_LENGTH * 2:], dtype=np.float32)
                    self.sample_update.emit(vis_data)
                
                # Run earthquake detection
                if len(buffer) >= INPUT_LENGTH:
                    if current_time - last_detection_time >= cooldown:
                        segment = np.array(list(buffer)[-INPUT_LENGTH:], dtype=np.float32)
                        is_eq, confidence = self.analyzer.detect_earthquake(segment)
                        
                        if is_eq:
                            wave_type, wave_conf = self.analyzer.classify_wave(segment)
                            magnitude = None
                            if wave_type == 'P':
                                magnitude, _ = self.analyzer.predict_magnitude(segment)
                            
                            det = WaveDetection(
                                wave_type=wave_type,
                                confidence=wave_conf,
                                start_sample=total_samples - INPUT_LENGTH,
                                end_sample=total_samples,
                                start_time=current_time - INPUT_LENGTH / self.sample_rate,
                                end_time=current_time,
                                magnitude=magnitude
                            )
                            self.detection.emit(det, current_time)
                            last_detection_time = current_time
        
        except ConnectionError:
            self.status_update.emit("⚠ Stream disconnected")
        except Exception as e:
            self.status_update.emit(f"⚠ Stream error: {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass
        
        self.status_update.emit("Stream monitoring complete")
        self.finished.emit()


# =============================================================================
# Custom Widgets
# =============================================================================
class WaveformCanvas(FigureCanvas):
    """Canvas for displaying seismic waveforms."""
    
    def __init__(self, parent=None):
        # Use the Slate 900 background for the figure
        self.fig = Figure(figsize=(10, 4), facecolor=COLORS['background'])
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLORS['background'])
        
        # Style ticks and labels with muted slate
        self.ax.tick_params(colors=COLORS['text_muted'], labelsize=9)
        
        # Style spines to blend in
        for spine in self.ax.spines.values():
            spine.set_edgecolor(COLORS['border'])
            spine.set_linewidth(1)
        
        # Add a subtle grid
        self.ax.grid(True, linestyle='--', alpha=0.1, color=COLORS['text'])
        
        self.fig.tight_layout()
    
    def plot_waveform(self, data: np.ndarray, sample_rate: float,
                      detections: List[WaveDetection] = None, title: str = ""):
        """Plot waveform with optional wave detections."""
        self.ax.clear()
        self.ax.set_facecolor(COLORS['background'])
        
        time_axis = np.arange(len(data)) / sample_rate
        # Use a vibrant Indigo for the waveform
        self.ax.plot(time_axis, data, color=COLORS['primary_hover'], linewidth=0.8, alpha=0.9)
        
        if detections:
            for d in detections:
                if d.wave_type != 'Noise':
                    # Determine color
                    color_key = d.wave_type.lower()
                    if 'p' in color_key: color = COLORS['p_wave']
                    elif 's' in color_key: color = COLORS['s_wave']
                    elif 'surface' in color_key: color = COLORS['surface_wave']
                    else: color = 'gray'
                    
                    # Arrival line (vibrant)
                    self.ax.axvline(x=d.start_time, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
                    
                    # Highlight span
                    self.ax.axvspan(
                        d.start_time, d.end_time,
                        alpha=0.15, color=color, label=d.wave_type
                    )
                    
                    # Label above the wave
                    y_max = np.max(data)
                    self.ax.text(d.start_time, y_max * 0.9, f" {d.wave_type}", 
                                color=color, fontweight='bold', fontsize=8)
        
        self.ax.set_xlabel('Time (seconds)', color=COLORS['text_muted'], fontweight='bold')
        self.ax.set_ylabel('Amplitude', color=COLORS['text_muted'], fontweight='bold')
        self.ax.grid(True, linestyle='--', alpha=0.1, color=COLORS['text'])
        self.ax.set_title(title, color=COLORS['primary_hover'], fontweight='bold', pad=15)
        self.ax.tick_params(colors=COLORS['text_muted'], labelsize=9)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_realtime(self, data: np.ndarray, sample_rate: float):
        """Plot real-time waveform."""
        self.ax.clear()
        self.ax.set_facecolor(COLORS['surface'])
        
        time_axis = np.arange(len(data)) / sample_rate
        self.ax.plot(time_axis, data, color='#00ff88', linewidth=1)
        
        self.ax.set_xlabel('Time (seconds)', color=COLORS['text'])
        self.ax.set_ylabel('Amplitude', color=COLORS['text'])
        self.ax.set_title('Real-time Waveform', color=COLORS['accent'], fontweight='bold')
        self.ax.tick_params(colors=COLORS['text'])
        self.ax.grid(True, alpha=0.2, color=COLORS['primary'])
        
        self.fig.tight_layout()
        self.draw()


class DetectionCard(QFrame):
    """Card widget displaying a wave detection."""
    
    def __init__(self, detection: WaveDetection, parent=None):
        super().__init__(parent)
        self.detection = detection
        self.setup_ui()
    
    def setup_ui(self):
        # Determine color based on wave type
        color_key = self.detection.wave_type.lower()
        if 'p' in color_key: color = COLORS['p_wave']
        elif 's' in color_key: color = COLORS['s_wave']
        elif 'surface' in color_key: color = COLORS['surface_wave']
        else: color = COLORS['text_muted']

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-left: 4px solid {color};
                border-radius: 10px;
                margin-bottom: 5px;
            }}
            QFrame:hover {{
                background-color: #2d3748;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        
        # Info Column
        info_layout = QVBoxLayout()
        
        type_label = QLabel(self.detection.wave_type.upper())
        type_label.setStyleSheet(f"font-weight: 900; font-size: 14px; color: {color};")
        info_layout.addWidget(type_label)
        
        time_range = QLabel(f"⏱ Arrival: {self.detection.start_time:.2f}s")
        time_range.setStyleSheet(f"color: {COLORS['text']}; font-size: 13px; font-weight: 600;")
        info_layout.addWidget(time_range)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Details Column
        details_layout = QVBoxLayout()
        details_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        conf_label = QLabel(f"Confidence: {self.detection.confidence:.1%}")
        conf_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        details_layout.addWidget(conf_label)
        
        if self.detection.magnitude is not None:
            mag_label = QLabel(f"M {self.detection.magnitude:.1f}")
            mag_label.setStyleSheet(f"color: {COLORS['warning']}; font-weight: 900; font-size: 14px;")
            details_layout.addWidget(mag_label)
            
        layout.addLayout(details_layout)


class AlertWidget(QFrame):
    """Floating toast notification for earthquake alerts."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setup_ui()
        
        # Opacity effect for animation
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        # Animations
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(400)
    
    def setup_ui(self):
        self.container = QFrame(self)
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 2px solid {COLORS['accent']};
                border-radius: 16px;
                padding: 15px;
            }}
        """)
        
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(20, 15, 20, 15)
        
        # Main layout for the widget
        self.widget_layout = QVBoxLayout(self)
        self.widget_layout.setContentsMargins(0, 0, 0, 0)
        self.widget_layout.addWidget(self.container)
        
        # Alert icon
        icon_label = QLabel("⚡")
        icon_label.setStyleSheet(f"font-size: 28px; color: {COLORS['accent']};")
        container_layout.addWidget(icon_label)
        
        # Alert text
        self.text_label = QLabel("EARTHQUAKE DETECTED!")
        self.text_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 800;
            color: {COLORS['text']};
            letter-spacing: 0.5px;
        """)
        container_layout.addWidget(self.text_label)
        
        container_layout.addSpacing(20)
        
        # Dismiss button
        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setFixedWidth(100)
        dismiss_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
        """)
        dismiss_btn.clicked.connect(self.hide_toast)
        container_layout.addWidget(dismiss_btn)
    
    def show_alert(self, wave_type: str, confidence: float, magnitude: float = None):
        """Show floating alert with detection details."""
        text = f"🚨 EARTHQUAKE: {wave_type}-wave detected ({confidence:.1%})"
        if magnitude is not None:
            text += f" | Magnitude: {magnitude:.1f}"
        self.text_label.setText(text)
        
        # Position at bottom right of parent
        if self.parent():
            # Get the parent's global position
            parent_pos = self.parent().mapToGlobal(self.parent().rect().topLeft())
            parent_rect = self.parent().rect()
            self.adjustSize()
            
            x = parent_pos.x() + parent_rect.width() - self.width() - 40
            y = parent_pos.y() + parent_rect.height() - self.height() - 40
            self.move(x, y)
        
        self.show()
        try:
            self.anim.finished.disconnect()
        except:
            pass
            
        self.anim.setStartValue(self.opacity_effect.opacity())
        self.anim.setEndValue(1.0)
        self.anim.start()
        
        # Auto-hide after 10 seconds
        QTimer.singleShot(10000, self.hide_toast)
        
    def hide_toast(self):
        if not self.isVisible():
            return
            
        try:
            self.anim.finished.disconnect()
        except:
            pass
            
        self.anim.finished.connect(self.hide)
        self.anim.setStartValue(self.opacity_effect.opacity())
        self.anim.setEndValue(0.0)
        self.anim.start()


# =============================================================================
# Main Application Window
# =============================================================================
class EarthquakeDetectorApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌍 Earthquake Detection System")
        self.setMinimumSize(1400, 900)
        
        # Initialize analyzer
        self.analyzer = None
        self.current_result = None
        self.current_data = None
        self.current_sample_rate = None
        self.realtime_worker = None
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()
        
        # Load AI models
        self.load_models()
    
    def setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Alert widget (floating toast)
        self.alert_widget = AlertWidget(self)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_analysis_tab()
        self.create_realtime_tab()
        self.create_results_tab()
        self.create_settings_tab()
    
    def create_analysis_tab(self):
        """Create the file analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File selection group
        file_group = QGroupBox("📁 File Selection")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 10px;")
        file_layout.addWidget(self.file_label, stretch=1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        
        self.analyze_btn = QPushButton("🔍 Analyze")
        self.analyze_btn.clicked.connect(self.analyze_file)
        self.analyze_btn.setEnabled(False)
        file_layout.addWidget(self.analyze_btn)
        
        layout.addWidget(file_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Splitter for waveform and results
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Waveform canvas
        waveform_group = QGroupBox("📈 Waveform Analysis")
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.analysis_canvas = WaveformCanvas()
        self.analysis_toolbar = NavigationToolbar(self.analysis_canvas, self)
        waveform_layout.addWidget(self.analysis_toolbar)
        waveform_layout.addWidget(self.analysis_canvas)
        
        splitter.addWidget(waveform_group)
        
        # Detection results
        results_group = QGroupBox("🎯 Detection Analysis")
        results_main_layout = QHBoxLayout(results_group)
        results_main_layout.setContentsMargins(15, 20, 15, 15)
        results_main_layout.setSpacing(20)
        
        # Left Window: Earthquake Event Summary
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"background-color: {COLORS['background']}; border-radius: 12px; border: 1px solid {COLORS['border']};")
        summary_layout = QVBoxLayout(summary_frame)
        
        summary_title = QLabel("EARTHQUAKE EVENT SUMMARY")
        summary_title.setStyleSheet(f"color: {COLORS['primary']}; font-weight: 900; font-size: 14px; letter-spacing: 1px;")
        summary_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_layout.addWidget(summary_title)
        
        self.result_summary = QLabel("Analyze a file to see earthquake metadata")
        self.result_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_summary.setStyleSheet("font-size: 14px; color: #94a3b8;")
        self.result_summary.setWordWrap(True)
        summary_layout.addWidget(self.result_summary, stretch=1)
        
        results_main_layout.addWidget(summary_frame, stretch=1)
        
        # Right Window: All Detected Waves
        waves_frame = QFrame()
        waves_frame.setStyleSheet(f"background-color: {COLORS['background']}; border-radius: 12px; border: 1px solid {COLORS['border']};")
        waves_layout = QVBoxLayout(waves_frame)
        
        waves_title = QLabel("PHASES & DETECTIONS")
        waves_title.setStyleSheet(f"color: {COLORS['primary']}; font-weight: 900; font-size: 14px; letter-spacing: 1px;")
        waves_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        waves_layout.addWidget(waves_title)
        
        # Detections scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.detections_widget = QWidget()
        self.detections_layout = QVBoxLayout(self.detections_widget) # Vertical for list
        self.detections_layout.setSpacing(10)
        self.detections_layout.addStretch()
        scroll.setWidget(self.detections_widget)
        waves_layout.addWidget(scroll)
        
        results_main_layout.addWidget(waves_frame, stretch=1)
        
        layout.addWidget(results_group)
        
        splitter.addWidget(results_group)
        splitter.setSizes([500, 300])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "📊 File Analysis")
    
    def create_realtime_tab(self):
        """Create the real-time monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Control panel
        control_group = QGroupBox("🎮 Monitoring Controls")
        control_layout = QHBoxLayout(control_group)
        
        # Source selection
        control_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Load from file", "Simulated data", "TCP Stream"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        control_layout.addWidget(self.source_combo)
        
        # TCP connection settings (hidden by default)
        self.tcp_host_label = QLabel("Host:")
        self.tcp_host_input = QComboBox()
        self.tcp_host_input.setEditable(True)
        self.tcp_host_input.addItems(["localhost", "127.0.0.1"])
        self.tcp_host_input.setCurrentText("localhost")
        self.tcp_host_input.setMaximumWidth(150)
        self.tcp_port_label = QLabel("Port:")
        self.tcp_port_input = QSpinBox()
        self.tcp_port_input.setRange(1024, 65535)
        self.tcp_port_input.setValue(9100)
        self.tcp_port_input.setMaximumWidth(80)
        
        control_layout.addWidget(self.tcp_host_label)
        control_layout.addWidget(self.tcp_host_input)
        control_layout.addWidget(self.tcp_port_label)
        control_layout.addWidget(self.tcp_port_input)
        
        # Initially hide TCP controls
        self.tcp_host_label.hide()
        self.tcp_host_input.hide()
        self.tcp_port_label.hide()
        self.tcp_port_input.hide()
        
        self.load_source_btn = QPushButton("Load Source")
        self.load_source_btn.clicked.connect(self.load_realtime_source)
        control_layout.addWidget(self.load_source_btn)
        
        control_layout.addStretch()
        
        # Speed control
        control_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(10)
        self.speed_slider.setMaximumWidth(150)
        self.speed_slider.valueChanged.connect(self.update_speed)
        control_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("1.0x")
        control_layout.addWidget(self.speed_label)
        
        control_layout.addStretch()
        
        # Start/Stop buttons
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        layout.addWidget(control_group)
        
        # Splitter for Waveform and Detections
        realtime_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Real-time waveform
        waveform_group = QGroupBox("📡 Live Waveform")
        waveform_layout = QVBoxLayout(waveform_group)
        self.realtime_canvas = WaveformCanvas()
        waveform_layout.addWidget(self.realtime_canvas)
        realtime_splitter.addWidget(waveform_group)
        
        # Phase Tracking & History
        tracking_group = QGroupBox("⚡ Active Event Tracking")
        tracking_main_layout = QHBoxLayout(tracking_group)
        
        # Left Side: Phase Status (P, S, Surface)
        self.phase_status_frame = QFrame()
        self.phase_status_frame.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 12px; border: 1px solid {COLORS['border']};")
        self.phase_status_frame.setFixedWidth(300)
        phase_status_layout = QVBoxLayout(self.phase_status_frame)
        
        phase_status_title = QLabel("PHASE ARRIVALS")
        phase_status_title.setStyleSheet(f"color: {COLORS['primary']}; font-weight: 900; font-size: 12px; letter-spacing: 1px;")
        phase_status_layout.addWidget(phase_status_title)
        
        # Phase items
        self.p_arrival_label = QLabel("P-Wave: --")
        self.s_arrival_label = QLabel("S-Wave: --")
        self.surface_arrival_label = QLabel("Surface: --")
        
        for label in [self.p_arrival_label, self.s_arrival_label, self.surface_arrival_label]:
            label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 14px; font-weight: bold; padding: 10px; background: {COLORS['background']}; border-radius: 6px; margin-top: 5px;")
            phase_status_layout.addWidget(label)
        
        phase_status_layout.addStretch()
        tracking_main_layout.addWidget(self.phase_status_frame)
        
        # Right Side: Detection history (vertical list)
        self.realtime_scroll = QScrollArea()
        self.realtime_scroll.setWidgetResizable(True)
        self.realtime_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.realtime_scroll.setStyleSheet("background: transparent;")
        
        self.realtime_cards_widget = QWidget()
        self.realtime_cards_layout = QVBoxLayout(self.realtime_cards_widget)
        self.realtime_cards_layout.setSpacing(8)
        self.realtime_cards_layout.addStretch()
        self.realtime_scroll.setWidget(self.realtime_cards_widget)
        tracking_main_layout.addWidget(self.realtime_scroll)
        
        realtime_splitter.addWidget(tracking_group)
        realtime_splitter.setSizes([600, 300])
        
        layout.addWidget(realtime_splitter)
        
        # Detection log (text)
        log_group = QGroupBox("📋 Detailed Log")
        log_layout = QVBoxLayout(log_group)
        
        self.detection_log = QTextEdit()
        self.detection_log.setReadOnly(True)
        self.detection_log.setMaximumHeight(150)
        log_layout.addWidget(self.detection_log)
        
        # Clear log button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_realtime_history)
        log_layout.addWidget(clear_btn)
        
        layout.addWidget(log_group)
        
        self.tab_widget.addTab(tab, "📡 Real-time Monitor")
    
    def create_results_tab(self):
        """Create the results and export tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results table
        table_group = QGroupBox("📊 Analysis History")
        table_layout = QVBoxLayout(table_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Earthquake", "Confidence", "Magnitude",
            "P-wave", "S-wave", "Surface"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table_layout.addWidget(self.results_table)
        
        layout.addWidget(table_group)
        
        # Export controls
        export_group = QGroupBox("💾 Export Results")
        export_layout = QHBoxLayout(export_group)
        
        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(lambda: self.export_results("json"))
        export_layout.addWidget(self.export_json_btn)
        
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export_results("csv"))
        export_layout.addWidget(self.export_csv_btn)
        
        export_layout.addStretch()
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        export_layout.addWidget(clear_history_btn)
        
        layout.addWidget(export_group)
        
        self.tab_widget.addTab(tab, "📋 Results")
    
    def create_settings_tab(self):
        """Create the settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model settings
        model_group = QGroupBox("🧠 AI Model Settings")
        model_layout = QFormLayout(model_group)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.99)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.05)
        model_layout.addRow("Detection Threshold:", self.threshold_spin)
        
        self.window_spin = QSpinBox()
        self.window_spin.setRange(100, 1000)
        self.window_spin.setValue(INPUT_LENGTH)
        model_layout.addRow("Window Size (samples):", self.window_spin)
        
        layout.addWidget(model_group)
        
        # Display settings
        display_group = QGroupBox("🎨 Display Settings")
        display_layout = QFormLayout(display_group)
        
        self.show_noise_check = QCheckBox()
        self.show_noise_check.setChecked(False)
        display_layout.addRow("Show Noise Segments:", self.show_noise_check)
        
        self.auto_alert_check = QCheckBox()
        self.auto_alert_check.setChecked(True)
        display_layout.addRow("Auto Alert on Detection:", self.auto_alert_check)
        
        layout.addWidget(display_group)
        
        # Model info
        info_group = QGroupBox("ℹ️ Model Information")
        info_layout = QVBoxLayout(info_group)
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(200)
        info_layout.addWidget(self.model_info_text)
        
        layout.addWidget(info_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "⚙️ Settings")
    
    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(lambda: self.export_results("json"))
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        
        analyze_action = QAction("&Analyze Current File", self)
        analyze_action.setShortcut("F5")
        analyze_action.triggered.connect(self.analyze_file)
        analysis_menu.addAction(analyze_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    

    def setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Model status
        self.model_status = QLabel("Loading models...")
        self.statusbar.addWidget(self.model_status)
        
        # Spacer
        self.statusbar.addPermanentWidget(QLabel(""))
        
        # GPU status
        self.gpu_status = QLabel("")
        self.statusbar.addPermanentWidget(self.gpu_status)
    
    def load_models(self):
        """Load the AI models."""
        try:
            self.model_status.setText("Loading AI models...")
            QApplication.processEvents()
            
            self.analyzer = SeismicAnalyzer(verbose=False)
            
            # Check which models loaded
            models_loaded = []
            if self.analyzer.earthquake_detector:
                models_loaded.append("Detector")
            if self.analyzer.wave_classifier:
                models_loaded.append("Classifier")
            if self.analyzer.magnitude_predictor:
                models_loaded.append("Magnitude")
            
            self.model_status.setText(f"✅ Models loaded: {', '.join(models_loaded)}")
            
            # Update model info
            info = f"Loaded Models:\n"
            info += f"- Earthquake Detector: {'✅' if self.analyzer.earthquake_detector else '❌'}\n"
            info += f"- Wave Classifier: {'✅' if self.analyzer.wave_classifier else '❌'}\n"
            info += f"- Magnitude Predictor: {'✅' if self.analyzer.magnitude_predictor else '❌'}\n"
            info += f"\nInput Window: {INPUT_LENGTH} samples ({INPUT_LENGTH/SAMPLE_RATE:.1f}s)\n"
            info += f"Sample Rate: {SAMPLE_RATE} Hz"
            self.model_info_text.setText(info)
            
            # Check GPU
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu_status.setText(f"🎮 GPU: {gpus[0].name.split('/')[-1]}")
            else:
                self.gpu_status.setText("💻 CPU Mode")
            
        except Exception as e:
            self.model_status.setText(f"❌ Error loading models: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load AI models:\n{str(e)}")
    
    def browse_file(self):
        """Open file browser dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Seismic Data File",
            "",
            "Seismic Files (*.mseed *.wav *.npy);;MiniSEED (*.mseed);;WAV Audio (*.wav);;NumPy (*.npy);;All Files (*)"
        )
        
        if filepath:
            self.current_filepath = filepath
            self.file_label.setText(f"📄 {Path(filepath).name}")
            self.analyze_btn.setEnabled(True)
    
    def analyze_file(self):
        """Analyze the selected file."""
        if not hasattr(self, 'current_filepath') or not self.analyzer:
            return
        
        self.analyze_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # Create worker thread
        self.analysis_worker = AnalysisWorker(self.analyzer, self.current_filepath)
        self.analysis_worker.progress.connect(self.update_progress)
        self.analysis_worker.finished.connect(self.analysis_complete)
        self.analysis_worker.error.connect(self.analysis_error)
        self.analysis_worker.start()
    
    def update_progress(self, value: int, message: str):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {message}")
    
    def analysis_complete(self, result_tuple):
        """Handle analysis completion."""
        result, data, sample_rate = result_tuple
        self.current_result = result
        self.current_data = data
        self.current_sample_rate = sample_rate
        
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)
        
        # Update waveform plot
        self.analysis_canvas.plot_waveform(
            data, sample_rate, result.detections,
            title=f"Analysis: {result.filename}"
        )
        
        # Update result summary
        if result.is_earthquake:
            summary = f"""
            <div style='margin-bottom: 20px;'>
                <h1 style='color: {COLORS["accent"]}; margin: 0; font-size: 24px;'>🚨 EARTHQUAKE DETECTED</h1>
                <p style='color: #94a3b8; font-size: 14px;'>Event confirmed by AI analysis engine.</p>
            </div>
            
            <div style='background-color: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155;'>
                <table style='width: 100%;'>
                    <tr>
                        <td style='color: #94a3b8;'>Confidence</td>
                        <td style='text-align: right; color: {COLORS["success"]}; font-weight: 800; font-size: 18px;'>{result.earthquake_confidence:.1%}</td>
                    </tr>
                    <tr><td colspan='2'><hr style='border: 0; border-top: 1px solid #334155;'></td></tr>
            """
            
            if result.estimated_magnitude:
                summary += f"""
                    <tr>
                        <td style='color: #94a3b8;'>Estimated Magnitude</td>
                        <td style='text-align: right; color: {COLORS["warning"]}; font-weight: 900; font-size: 24px;'>{result.estimated_magnitude:.1f} M<sub>w</sub></td>
                    </tr>
                    <tr><td colspan='2'><hr style='border: 0; border-top: 1px solid #334155;'></td></tr>
                """
            
            summary += f"""
                    <tr>
                        <td style='color: #94a3b8;'>P-wave Arrival</td>
                        <td style='text-align: right; color: {COLORS["p_wave"]}; font-weight: bold;'>{result.p_wave_arrival:.2f}s</td>
                    </tr>
                    <tr>
                        <td style='color: #94a3b8;'>S-wave Arrival</td>
                        <td style='text-align: right; color: {COLORS["s_wave"]}; font-weight: bold;'>{result.s_wave_arrival:.2f}s</td>
                    </tr>
                    <tr>
                        <td style='color: #94a3b8;'>Surface Wave</td>
                        <td style='text-align: right; color: {COLORS["surface_wave"]}; font-weight: bold;'>{result.surface_wave_arrival:.2f}s</td>
                    </tr>
                </table>
            </div>
            """
        else:
            summary = f"""
            <div style='text-align: center; padding: 40px;'>
                <div style='font-size: 48px; margin-bottom: 20px;'>✅</div>
                <h2 style='color: {COLORS["success"]}; margin: 0;'>No Earthquake Detected</h2>
                <p style='color: #94a3b8; font-size: 14px;'>Signal classified as background noise.</p>
            </div>
            """
        
        summary += f"<div style='margin-top: 20px; text-align: center; color: #64748b; font-size: 11px;'>Processing time: {result.processing_time:.3f}s</div>"
        self.result_summary.setText(summary)
        
        # Show alert if configured
        if result.is_earthquake and self.auto_alert_check.isChecked():
            self.alert_widget.show_alert(
                "P" if result.p_wave_arrival else "S" if result.s_wave_arrival else "Surface",
                result.earthquake_confidence,
                result.estimated_magnitude
            )
        
        # Update detections cards
        self.update_detection_cards(result.detections)
        
        # Add to results table
        self.add_to_results_table(result)
        
        self.statusbar.showMessage(f"Analysis complete: {result.filename}", 5000)
    
    def update_detection_cards(self, detections: List[WaveDetection]):
        """Update the detection cards display."""
        # Clear existing cards
        while self.detections_layout.count():
            child = self.detections_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add new cards
        earthquake_detections = [d for d in detections if d.wave_type != 'Noise']
        
        for detection in earthquake_detections[:10]:  # Limit to 10 cards
            card = DetectionCard(detection)
            self.detections_layout.addWidget(card)
        
        self.detections_layout.addStretch()
    
    def add_to_results_table(self, result: AnalysisResult):
        """Add analysis result to the history table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(result.filename))
        self.results_table.setItem(row, 1, QTableWidgetItem("Yes" if result.is_earthquake else "No"))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.earthquake_confidence:.1%}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(
            f"{result.estimated_magnitude:.1f}" if result.estimated_magnitude else "-"
        ))
        self.results_table.setItem(row, 4, QTableWidgetItem(
            f"{result.p_wave_arrival:.2f}s" if result.p_wave_arrival else "-"
        ))
        self.results_table.setItem(row, 5, QTableWidgetItem(
            f"{result.s_wave_arrival:.2f}s" if result.s_wave_arrival else "-"
        ))
        self.results_table.setItem(row, 6, QTableWidgetItem(
            f"{result.surface_wave_arrival:.2f}s" if result.surface_wave_arrival else "-"
        ))
    
    def analysis_error(self, error_message: str):
        """Handle analysis error."""
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"Error analyzing file:\n{error_message}")
    
    def _on_source_changed(self, text: str):
        """Show/hide TCP controls based on source selection."""
        is_tcp = (text == "TCP Stream")
        self.tcp_host_label.setVisible(is_tcp)
        self.tcp_host_input.setVisible(is_tcp)
        self.tcp_port_label.setVisible(is_tcp)
        self.tcp_port_input.setVisible(is_tcp)
        # For TCP, load_source just validates; for others, it loads data
        if is_tcp:
            self.load_source_btn.setText("Connect")
        else:
            self.load_source_btn.setText("Load Source")
    
    def load_realtime_source(self):
        """Load data source for real-time monitoring."""
        source = self.source_combo.currentText()
        
        if source == "Load from file":
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Select Source File",
                "",
                "Seismic Files (*.mseed *.wav *.npy);;All Files (*)"
            )
            
            if filepath:
                try:
                    self.realtime_data, self.realtime_sr = self.analyzer.load_file(filepath)
                    self.realtime_source_type = "file"
                    self.start_btn.setEnabled(True)
                    self.statusbar.showMessage(f"Loaded source: {Path(filepath).name}", 3000)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
        
        elif source == "TCP Stream":
            # For TCP, we just mark it ready; actual connection happens on start
            self.realtime_source_type = "tcp"
            self.tcp_stream_host = self.tcp_host_input.currentText().strip()
            self.tcp_stream_port = self.tcp_port_input.value()
            self.start_btn.setEnabled(True)
            self.statusbar.showMessage(
                f"TCP stream configured: {self.tcp_stream_host}:{self.tcp_stream_port} "
                f"– press Start to connect", 5000
            )
        
        else:
            # Generate simulated data
            duration = 60  # seconds
            self.realtime_sr = SAMPLE_RATE
            self.realtime_data = np.random.randn(int(duration * SAMPLE_RATE)).astype(np.float32) * 0.1
            self.realtime_source_type = "simulated"
            self.start_btn.setEnabled(True)
            self.statusbar.showMessage("Loaded simulated noise data", 3000)
    
    def update_speed(self, value):
        """Update playback speed."""
        speed = value / 10.0
        self.speed_label.setText(f"{speed:.1f}x")
        if self.realtime_worker:
            self.realtime_worker.set_speed(speed)
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        source_type = getattr(self, 'realtime_source_type', None)
        
        if source_type == "tcp":
            # Use TCP stream worker
            host = getattr(self, 'tcp_stream_host', 'localhost')
            port = getattr(self, 'tcp_stream_port', 9100)
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.load_source_btn.setEnabled(False)
            
            self.realtime_worker = TcpStreamWorker(
                self.analyzer, host, port
            )
            self.realtime_sr = SAMPLE_RATE
            self.realtime_worker.detection.connect(self.handle_realtime_detection)
            self.realtime_worker.sample_update.connect(self.update_realtime_plot)
            self.realtime_worker.status_update.connect(self._handle_stream_status)
            self.realtime_worker.finished.connect(self.monitoring_finished)
            # Reset phase status labels
            self.p_arrival_label.setText("P-Wave: --")
            self.s_arrival_label.setText("S-Wave: --")
            self.surface_arrival_label.setText("Surface: --")
            self.p_arrival_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 14px; font-weight: bold; padding: 10px; background: {COLORS['background']}; border-radius: 6px; margin-top: 5px;")
            self.s_arrival_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 14px; font-weight: bold; padding: 10px; background: {COLORS['background']}; border-radius: 6px; margin-top: 5px;")
            self.surface_arrival_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 14px; font-weight: bold; padding: 10px; background: {COLORS['background']}; border-radius: 6px; margin-top: 5px;")
            
            self.realtime_worker.start()
            
            self.detection_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Connecting to TCP stream at {host}:{port}..."
            )
        else:
            # File-based or simulated monitoring
            if not hasattr(self, 'realtime_data'):
                return
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.load_source_btn.setEnabled(False)
            
            self.realtime_worker = RealtimeWorker(
                self.analyzer, self.realtime_data, self.realtime_sr
            )
            self.realtime_worker.set_speed(self.speed_slider.value() / 10.0)
            self.realtime_worker.detection.connect(self.handle_realtime_detection)
            self.realtime_worker.sample_update.connect(self.update_realtime_plot)
            self.realtime_worker.status_update.connect(lambda s: self.statusbar.showMessage(s))
            self.realtime_worker.finished.connect(self.monitoring_finished)
            self.realtime_worker.start()
            
            self.detection_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring started...")
    
    def _handle_stream_status(self, message: str):
        """Handle TCP stream status updates in both statusbar and detection log."""
        self.statusbar.showMessage(message)
        self.detection_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if self.realtime_worker:
            self.realtime_worker.stop()
    
    def monitoring_finished(self):
        """Handle monitoring completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_source_btn.setEnabled(True)
        self.detection_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring stopped")
    
    def handle_realtime_detection(self, detection: WaveDetection, timestamp: float):
        """Handle real-time detection and update the phase-specific UI components."""
        time_str = datetime.now().strftime('%H:%M:%S')
        wave_type = detection.wave_type
        color_key = wave_type.lower()
        
        # 1. Update the dedicated Phase Status readouts
        if 'p' in color_key:
            color = COLORS['p_wave']
            self.p_arrival_label.setText(f"P-Wave: {timestamp:.2f}s")
            self.p_arrival_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 900; padding: 10px; background: {COLORS['surface']}; border: 1px solid {color}; border-radius: 6px; margin-top: 5px;")
        elif 's' in color_key:
            color = COLORS['s_wave']
            self.s_arrival_label.setText(f"S-Wave: {timestamp:.2f}s")
            self.s_arrival_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 900; padding: 10px; background: {COLORS['surface']}; border: 1px solid {color}; border-radius: 6px; margin-top: 5px;")
        elif 'surface' in color_key:
            color = COLORS['surface_wave']
            self.surface_arrival_label.setText(f"Surface: {timestamp:.2f}s")
            self.surface_arrival_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 900; padding: 10px; background: {COLORS['surface']}; border: 1px solid {color}; border-radius: 6px; margin-top: 5px;")
        else:
            color = COLORS['text_muted']

        # 2. Log the detection with thematic coloring
        log_msg = f"[{time_str}] 🚨 {wave_type}-wave detected ({detection.confidence:.1%})"
        if detection.magnitude:
            log_msg += f" | Magnitude: {detection.magnitude:.1f}"
        self.detection_log.append(f'<span style="color: {color}; font-weight: bold;">{log_msg}</span>')
        
        # 3. Show floating toast notification
        if self.auto_alert_check.isChecked():
            self.alert_widget.show_alert(wave_type, detection.confidence, detection.magnitude)
        
        # 4. Add visual card to the vertical detections list
        card = DetectionCard(detection)
        self.realtime_cards_layout.insertWidget(0, card) # Most recent at top
        
        # Limit history cards to 15 for performance
        if self.realtime_cards_layout.count() > 16: # +1 for stretch
            item = self.realtime_cards_layout.takeAt(self.realtime_cards_layout.count() - 2)
            if item and item.widget():
                item.widget().deleteLater()
        
        # 5. Record in the main results table
        source_name = f"Live: {self.source_combo.currentText()}"
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(source_name))
        self.results_table.setItem(row, 1, QTableWidgetItem("Yes"))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{detection.confidence:.1%}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{detection.magnitude:.1f}" if detection.magnitude else "-"))
        
        # Initialize wave columns
        for i in range(4, 7):
            self.results_table.setItem(row, i, QTableWidgetItem("-"))
        
        # Put timestamp in the correct wave column
        col_map = {"P": 4, "S": 5, "Surface": 6}
        if wave_type in col_map:
            self.results_table.setItem(row, col_map[wave_type], QTableWidgetItem(f"{timestamp:.2f}s"))
            
        self.statusbar.showMessage(f"Real-time {wave_type}-wave detected!", 3000)
    
    def clear_realtime_history(self):
        """Clear the real-time detection history (cards and log)."""
        self.detection_log.clear()
        while self.realtime_cards_layout.count():
            child = self.realtime_cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def update_realtime_plot(self, data: np.ndarray):
        """Update real-time waveform plot."""
        self.realtime_canvas.plot_realtime(data, self.realtime_sr)
    
    def export_results(self, format: str):
        """Export results to file."""
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "No Data", "No results to export.")
            return
        
        if format == "json":
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export JSON", "", "JSON Files (*.json)"
            )
            if filepath:
                results = []
                for row in range(self.results_table.rowCount()):
                    results.append({
                        'file': self.results_table.item(row, 0).text(),
                        'earthquake': self.results_table.item(row, 1).text(),
                        'confidence': self.results_table.item(row, 2).text(),
                        'magnitude': self.results_table.item(row, 3).text(),
                        'p_wave': self.results_table.item(row, 4).text(),
                        's_wave': self.results_table.item(row, 5).text(),
                        'surface_wave': self.results_table.item(row, 6).text(),
                    })
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2)
                self.statusbar.showMessage(f"Exported to {filepath}", 3000)
        
        elif format == "csv":
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV Files (*.csv)"
            )
            if filepath:
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['File', 'Earthquake', 'Confidence', 'Magnitude',
                                    'P-wave', 'S-wave', 'Surface Wave'])
                    for row in range(self.results_table.rowCount()):
                        writer.writerow([
                            self.results_table.item(row, col).text()
                            for col in range(7)
                        ])
                self.statusbar.showMessage(f"Exported to {filepath}", 3000)
    
    def clear_history(self):
        """Clear results history."""
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to clear all results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.results_table.setRowCount(0)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Earthquake Detection System",
            """<h2>🌍 Earthquake Detection System</h2>
            <p>Version 1.0</p>
            <p>A desktop application for real-time earthquake detection
            and seismic wave classification using AI.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Earthquake vs Noise detection (96.8% accuracy)</li>
                <li>P, S, Surface wave classification (99.7% accuracy)</li>
                <li>Magnitude prediction from P-waves</li>
                <li>Real-time monitoring</li>
                <li>Support for .mseed, .wav, .npy files</li>
            </ul>
            <p>Powered by TensorFlow and PyQt6</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.realtime_worker and self.realtime_worker.isRunning():
            self.realtime_worker.stop()
            self.realtime_worker.wait()
        event.accept()


# =============================================================================
# Entry Point
# =============================================================================
def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    
    # Set application info
    app.setApplicationName("Earthquake Detection System")
    app.setOrganizationName("SeismicQuake")
    
    # Create and show main window
    window = EarthquakeDetectorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
