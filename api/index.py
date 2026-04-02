import sys
import os

# Add project root to sys.path so app.py (and its templates/ folder) are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401  – Vercel picks up the WSGI `app` object
