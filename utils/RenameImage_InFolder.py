'''
txt_folder_all_dataからtxt_all_dataを作成する。
ディレクトリ内にあるすべての.txtファイルを収集し、新しいディレクトリにコピーする。
'''

import os  # OS操作用モジュール
import shutil  # ファイルコピー用モジュール

def copy_all_txt_files(source_dir, destination_dir):
    """
    指定フォルダ内の全ての.txtファイルを再帰的に探し、コピーする。

    :param source_dir: 元のフォルダのパス
    :param destination_dir: コピー先のフォルダのパス
    """
    os.makedirs(destination_dir, exist_ok=True)  # コピー先フォルダがなければ作成

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt'):  # .txtファイルのみを対象
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)

                shutil.copy2(source_file_path, destination_file_path)  # ファイルをコピー
                print(f"Copied: {source_file_path} to {destination_file_path}")

# 使用例
source_directory = '/Users/nagasawa/Downloads/GraduationThesis/New_txt_all_data'
destination_directory = '/Users/nagasawa/Downloads/GraduationThesis/txt_all_data'
copy_all_txt_files(source_directory, destination_directory)
