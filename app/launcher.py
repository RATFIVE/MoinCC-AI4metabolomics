import subprocess
import platform

# Example: Listing files in the current directory (similar to 'dir')
command = r"streamlit run D:\\Dokumente\\MoinCC-AI4metabolomics\\app\\app.py"  # Windows command
result = subprocess.run(command, shell=True, text=True, capture_output=True)

# Print the output and any error messages
print("Output:")
print(result.stdout)
print("Errors:")
print(result.stderr)
