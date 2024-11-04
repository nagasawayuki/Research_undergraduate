'''
txt_folder_all_dataからtxt_all_dataを作成する
ディレクトリの中にディレクトリがあって、その中にtxtファイルがある場合、
中間のディレクトリを除いた新しいディレクトリを作ってる
'''

import os  # OS操作のためのモジュールをインポート
import shutil  # ファイルのコピーのためのモジュールをインポート

def copy_all_txt_files(source_dir, destination_dir):
    """
    指定されたフォルダとそのサブフォルダ内にあるすべての .txt ファイルを指定した出力フォルダにコピーする。

    :param source_dir: 元のフォルダのパス
    :param destination_dir: コピー先のフォルダのパス
    """
    # コピー先のフォルダが存在しない場合は作成する
    os.makedirs(destination_dir, exist_ok=True)

    '''
 /path/to/directory
├── file1.txt
├── file2.txt
├── subdir1
│   ├── file3.txt
│   └── file4.txt
└── subdir2
    └── file5.txt

Current directory: /path/to/directory
Subdirectories: ['subdir1', 'subdir2']
Files: ['file1.txt', 'file2.txt']

Current directory: /path/to/directory/subdir1
Subdirectories: []
Files: ['file3.txt', 'file4.txt']

Current directory: /path/to/directory/subdir2
Subdirectories: []
Files: ['file5.txt']
    '''
    # 元フォルダ内のディレクトリとファイルを再帰的に探索
    for root, dirs, files in os.walk(source_dir):
        for file in files:  # 各ディレクトリ内のファイルを順番に確認
            # .txtファイルのみを対象とする
            if file.endswith('.txt'):
                '''
                os.path.join→パスの生成

folder_path = '/Users/nagasawa/Documents'  # フォルダのパス
file_name = 'data.txt'  # ファイル名

# フォルダパスとファイル名を連結してフルパスを作成
full_path = os.path.join(folder_path, file_name)  ー＞ /Users/nagasawa/Documents/data.txt
                '''
                source_file_path = os.path.join(root, file)  # 元ファイルのパス
                destination_file_path = os.path.join(destination_dir, file)  # コピー先のファイルパス

                # ファイルをコピーする
                shutil.copy2(source_file_path, destination_file_path)
                print(f"Copied: {source_file_path} to {destination_file_path}")  # コピー完了メッセージを表示

# 使用例
source_directory = '/Users/nagasawa/Downloads/GraduationThesis/New_txt_all_data'  # コピー元フォルダのパスを指定
destination_directory = '/Users/nagasawa/Downloads/GraduationThesis/txt_all_data'  # コピー先フォルダのパスを指定
copy_all_txt_files(source_directory, destination_directory)  # 関数を実行してテキストファイルをコピー
