# Optional post-install hook for ComfyUI Manager.
# Installs requirements.txt if present.

import os
import sys
import subprocess

def main():
    here = os.path.dirname(__file__)
    req = os.path.join(here, "requirements.txt")
    if os.path.exists(req):
        # If requirements are empty, this will no-op quickly.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])

if __name__ == "__main__":
    main()
