import os
import glob

# 親ディレクトリのパスを指定（このディレクトリ内の各サブディレクトリ名を変更します）
parent_directory = '/Users/nagasawa/Downloads/content 5/yolov5/runs/detect'

# 親ディレクトリ内のすべてのサブディレクトリを取得
subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

for subdir in subdirectories:
    # サブディレクトリ内のすべての.mp4ファイルを取得
    mp4_files = glob.glob(os.path.join(parent_directory, subdir, '*.mp4'))
    
    # .mp4ファイルが正確に一つだけあることを確認
    if len(mp4_files) == 1:
        # ファイル名から.mp4を除いた名前を取得
        new_dir_name = os.path.basename(mp4_files[0]).replace('.mp4', '')
        
        # 現在のディレクトリパス
        current_dir_path = os.path.join(parent_directory, subdir)
        
        # 新しいディレクトリパス
        new_dir_path = os.path.join(parent_directory, new_dir_name)
        
        # ディレクトリ名を変更
        os.rename(current_dir_path, new_dir_path)
        print(f'Renamed "{current_dir_path}" to "{new_dir_path}"')
    else:
        print(f'Skipped "{subdir}" because it does not contain exactly one .mp4 file.')
