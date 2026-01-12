import os
import sys
from pathlib import Path

try:
    import pypdf
except ImportError:
    print("pypdf not installed. Please install it with 'pip install pypdf'")
    sys.exit(1)

def extract_text_from_pdf(pdf_path):
    print(f"--- Extracting from {pdf_path.name} ---")
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n[Page {i+1}]\n"
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading {pdf_path.name}: {e}"

base_path = Path(r"d:\projects\seismicquake\References\ppt")
files = [
    "Earthquake[zeroth].pdf",
    "Seismic Detection And Visualisation and Analysis(Final).pdf"
]

for filename in files:
    path = base_path / filename
    if path.exists():
        content = extract_text_from_pdf(path)
        print(content[:2000]) # Print first 2000 chars to avoid huge output
        print("\n... [truncated] ...\n")
        # Check for specific keywords
        keywords = ["Existing System", "Proposed System", "Methodology", "Algorithm", "Conclusion"]
        print(f"--- Keywords in {filename} ---")
        lower_content = content.lower()
        for kw in keywords:
            if kw.lower() in lower_content:
                print(f"Found '{kw}'")
                # Try to print context
                idx = lower_content.find(kw.lower())
                start = max(0, idx - 100)
                end = min(len(content), idx + 500)
                print(f"Context: ...{content[start:end].replace(chr(10), ' ')}...")
    else:
        print(f"File not found: {path}")
