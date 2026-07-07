import subprocess
import sys
import shutil
import time

def run_services():
    # Use shutil.which to find the absolute path of the executables in the current path
    uvicorn_path = shutil.which("uvicorn")
    streamlit_path = shutil.which("streamlit")

    if not uvicorn_path or not streamlit_path:
        print("Error: Could not find uvicorn or streamlit in the PATH.")
        print("Please ensure your Conda environment is activated.")
        return

    api_cmd = [uvicorn_path, "api.app:app", "--reload", "--port", "8000"]
    streamlit_cmd = [streamlit_path, "run", "streamlit/app.py", "--server.port", "8501"]
    
    print(f"Launching API and Streamlit...")
    
    # Use subprocess.Popen with shell=False for security and stability
    api_process = subprocess.Popen(api_cmd)
    
    print("Waiting 10 seconds for API to initialize...")
    time.sleep(10) 
    
    streamlit_process = subprocess.Popen(streamlit_cmd)

    try:
        # Keep the main process alive to monitor the children
        while api_process.poll() is None and streamlit_process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        api_process.terminate()
        streamlit_process.terminate()
        # Ensure processes are cleaned up
        api_process.wait()
        streamlit_process.wait()

if __name__ == "__main__":
    run_services()