#指定された親フォルダ内のすべてのサブフォルダに含まれる画像ファイルの名前を変更するためのもの
#各画像ファイルは、そのサブフォルダの名前に基づいて新しい名前が付けられる
#movie_to_image.pyを実行した後に有効なプログラム

import os

def rename_images_based_on_parent_folder(parent_folder):
    """
    Renames all image files in each subfolder of the given parent folder.
    The new file names will include the subfolder name followed by an underscore and numbering.

    :param parent_folder: Path to the parent folder containing subfolders with images.
    """
    for folder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for i, filename in enumerate(os.listdir(subfolder_path)):
                if filename.endswith(".png"):
                    new_filename = f"{folder_name}_{i:03}.png"
                    os.rename(os.path.join(subfolder_path, filename), os.path.join(subfolder_path, new_filename))
                    print(f"Renamed {filename} to {new_filename} in folder {folder_name}")

# Example usage:
rename_images_based_on_parent_folder("/Users/nagasawa/Downloads/Prepareing_DL_dataset/データセット/extracted_frames")

# Note: Replace "path/to/dataset" with the actual path to your parent folder containing the image subfolders.

