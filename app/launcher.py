import subprocess
import platform

# Path to your Streamlit app
app_file = "app.py"

# Determine the platform and open the terminal accordingly
def start_streamlit():
    try:
        if platform.system() == "Windows":
            # For Windows, use 'start' to open a new terminal
            subprocess.run(f'start cmd /k streamlit run {app_file}', shell=True)
        elif platform.system() == "Darwin":  # macOS
            # For macOS, use 'open -a Terminal' to open a new terminal
            subprocess.run(f'open -a Terminal "streamlit run {app_file}"', shell=True)
        elif platform.system() == "Linux":
            # For Linux, use 'gnome-terminal' or another terminal emulator
            subprocess.run(f'gnome-terminal -- bash -c "streamlit run {app_file}; exec bash"', shell=True)
        else:
            print("Unsupported operating system.")
    except Exception as e:
        print(f"Failed to start Streamlit: {e}")

# Run the function
start_streamlit()
