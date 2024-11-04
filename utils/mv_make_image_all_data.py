import os
import shutil

def copy_all_png_files(source_dir, destination_dir):
    """
    指定フォルダ内のすべての.pngファイルをコピーする。

    :param source_dir: 元のフォルダのパス
    :param destination_dir: コピー先のフォルダのパス
    """
    os.makedirs(destination_dir, exist_ok=True)  # コピー先フォルダを作成

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png'):  # .pngファイルのみ対象
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)

                shutil.copy2(source_file_path, destination_file_path)  # ファイルをコピー
                print(f"Copied: {source_file_path} to {destination_file_path}")

# 使用例
source_directory = '/Users/nagasawa/Downloads/GraduationThesis/New_Dataset/image_folder_all_data'
destination_directory = '/Users/nagasawa/Downloads/GraduationThesis/New_Dataset/image_all_data'
copy_all_png_files(source_directory, destination_directory)
