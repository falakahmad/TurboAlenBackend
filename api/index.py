"""
Vercel serverless function entry point for FastAPI app.
This file is used by Vercel to serve the FastAPI application.
"""
import sys
import os

# Get the directory containing this file (api/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (backend/)
backend_dir = os.path.dirname(current_dir)

# Add both directories to Python path for imports
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the FastAPI app
try:
    from api.main import app
except ImportError:
    # Fallback: try importing from main directly if we're in the api directory
    from main import app

# Export the app for Vercel
__all__ = ["app"]
