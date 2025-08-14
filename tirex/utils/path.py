from pathlib import Path


def cleanup_directory(directory_path):
    # Create a Path object for the directory
    path = Path(directory_path)

    # Ensure the directory exists
    if not path.exists() or not path.is_dir():
        print(f"The directory {directory_path} does not exist.")
        return

    # Iterate through each item in the directory
    for item in path.iterdir():
        # Check if it's a file and delete it
        if item.is_file():
            try:
                item.unlink()  # Delete the file
                print(f"Deleted file: {item}")
            except Exception as e:
                print(f"Error deleting file {item}: {e}")
