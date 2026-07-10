import sys
import os

# Add src/bookfriend to Python path so pytest can find database, api, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))