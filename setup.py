import os
import subprocess
import platform

def install_ffmpeg():
    """
    Install ffmpeg based on the operating system.
    """
    os_name = platform.system()
    try:
        if os_name == "Linux":
            print("Installing ffmpeg on Linux...")
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        elif os_name == "Darwin":
            print("Installing ffmpeg on macOS...")
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
        elif os_name == "Windows":
            print("Please install ffmpeg manually on Windows. Download it from https://ffmpeg.org/download.html")
        else:
            print(f"Unsupported operating system: {os_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing ffmpeg: {e}")
    else:
        print("ffmpeg installed successfully (if supported).")

def create_folders():
    """
    Create necessary folders under the 'static' directory.
    """
    base_dir = os.path.join(os.getcwd(), "static")
    folders = ["frames", "rendered_frames", "uploads"]

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder created: {folder_path}")

def main():
    print("Starting setup...")
    
    # Install ffmpeg
    install_ffmpeg()
    
    # Create necessary folders
    create_folders()

    print("Setup completed successfully!")

if __name__ == "__main__":
    main()
