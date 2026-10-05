"""
Single Entry Point Runner for Decoupled Visual Defect Detection System
Starts the FastAPI Backend REST Server (port 8000) and launches the Streamlit Frontend Client (port 8501) in a single command.
"""

import subprocess
import time
import sys
import os
import socket
import urllib.request

# Ensure UTF-8 output encoding for Windows compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def is_port_in_use(port: int) -> bool:
    """Checks if a local TCP port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def kill_port_process(port: int):
    """Frees local port by killing process using netstat & taskkill on Windows."""
    try:
        if sys.platform.startswith('win'):
            cmd = f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :{port} ^| findstr LISTENING\') do taskkill /F /PID %a'
            subprocess.run(cmd, shell=True, capture_output=True)
    except Exception:
        pass

def wait_for_backend(url="http://localhost:8000/health", timeout=15):
    """Waits for FastAPI backend server to report healthy status."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def main():
    print("=" * 75)
    print(" STARTING GLASS BOTTLE ANOMALY DETECTION & QUALITY TRIAGE SYSTEM ")
    print("=" * 75)

    # Free port 8000 if occupied by old process
    if is_port_in_use(8000):
        print("[*] Freeing occupied port 8000...", flush=True)
        kill_port_process(8000)
        time.sleep(1)

    print("1. Launching FastAPI REST API Backend Server on http://localhost:8000 ...", flush=True)

    # Launch FastAPI backend process
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    backend_proc = subprocess.Popen(backend_cmd)

    print("[*] Waiting for backend server to initialize...", flush=True)
    if wait_for_backend():
        print("[+] Backend API is ONLINE! (Docs: http://localhost:8000/docs)", flush=True)
    else:
        print("[!] Warning: Backend server taking longer than expected. Continuing...", flush=True)

    # Choose available port for Streamlit (8501, 8502, 8503, etc.)
    frontend_port = 8501
    while is_port_in_use(frontend_port):
        print(f"[*] Port {frontend_port} is busy. Attempting port cleanup...", flush=True)
        kill_port_process(frontend_port)
        time.sleep(1)
        if is_port_in_use(frontend_port):
            frontend_port += 1

    print(f"\n2. Launching Streamlit Web Frontend Client on http://localhost:{frontend_port} ...", flush=True)
    
    # Launch Streamlit frontend process
    frontend_cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(frontend_port)]
    frontend_proc = subprocess.Popen(frontend_cmd)
    
    try:
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Stopping application servers...")
    finally:
        try:
            frontend_proc.terminate()
        except Exception:
            pass
        try:
            backend_proc.terminate()
        except Exception:
            pass
        print("[+] Shutdown complete.", flush=True)

if __name__ == "__main__":
    main()
