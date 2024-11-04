import os
import glob

# 指定フォルダ内の各サブディレクトリの名前をリネームするスクリプト
parent_directory = '/Users/nagasawa/Downloads/content 5/yolov5/runs/detect'

# サブディレクトリを取得
subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

for subdir in subdirectories:
    # .mp4ファイルを取得
    mp4_files = glob.glob(os.path.join(parent_directory, subdir, '*.mp4'))
    
    if len(mp4_files) == 1:  # .mp4ファイルが1つだけの場合
        new_dir_name = os.path.basename(mp4_files[0]).replace('.mp4', '')
        
        current_dir_path = os.path.join(parent_directory, subdir)
        new_dir_path = os.path.join(parent_directory, new_dir_name)
        
        os.rename(current_dir_path, new_dir_path)
        print(f'Renamed "{current_dir_path}" to "{new_dir_path}"')
    else:
        print(f'Skipped "{subdir}" because it does not contain exactly one .mp4 file.')

