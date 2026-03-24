import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so test runner can import the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Provide a default FINNHUB_API_KEY for tests so importing the package doesn't fail
os.environ.setdefault("FINNHUB_API_KEY", "test")
