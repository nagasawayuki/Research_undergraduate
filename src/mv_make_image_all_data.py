import os
import shutil

def copy_all_png_files(source_dir, destination_dir):
    """
    Copy all .png files from source_dir and its subdirectories to destination_dir.

    :param source_dir: Path to the source directory.
    :param destination_dir: Path to the destination directory.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png'):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)

                # Copy the file
                shutil.copy2(source_file_path, destination_file_path)
                print(f"Copied: {source_file_path} to {destination_file_path}")

# Usage
source_directory = '/Users/nagasawa/Downloads/GraduationThesis/New_Dataset/image_folder_all_data'  # Replace with the actual source directory path
destination_directory = '/Users/nagasawa/Downloads/GraduationThesis/New_Dataset/image_all_data'  # Replace with the actual destination directory path
copy_all_png_files(source_directory, destination_directory)