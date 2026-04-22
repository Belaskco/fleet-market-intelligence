import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.app_interface import run_dashboard

if __name__ == "__main__":
    run_dashboard()