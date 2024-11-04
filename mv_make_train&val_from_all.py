import os
import shutil
import random

# 指定されたフォルダのパス
image_all_data_folder = '/Users/nagasawa/Downloads/GraduationThesis/New_Dataset/image_all_data'
txt_all_data_folder = '/Users/nagasawa/Downloads/GraduationThesis/New_Dataset/txt_all_data'

# 新しいフォルダを作成（存在しない場合のみ）
def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

create_folder_if_not_exists('image_train')
create_folder_if_not_exists('image_val')
create_folder_if_not_exists('txt_train')
create_folder_if_not_exists('txt_val')

# ファイルのリストを取得し、拡張子を除いた名前でソート
image_files = sorted([f for f in os.listdir(image_all_data_folder) if f.endswith('.png')])
txt_files = sorted([f for f in os.listdir(txt_all_data_folder) if f.endswith('.txt')])

# ファイル名（拡張子を除く）が一致するか確認
assert len(image_files) == len(txt_files) and all([os.path.splitext(image)[0] == os.path.splitext(txt)[0] for image, txt in zip(image_files, txt_files)]), "画像とテキストのファイル名が一致しません。"

# ファイルをランダムにシャッフル
combined = list(zip(image_files, txt_files))
random.shuffle(combined)

# トレーニングセットとバリデーションセットに分割（8:2の比率）
split_index = int(len(combined) * 0.8)
train_files = combined[:split_index]
val_files = combined[split_index:]

# ファイルを対応するフォルダにコピー
def copy_files(files, source_folder_image, source_folder_txt, target_folder_image, target_folder_txt):
    for image_file, txt_file in files:
        shutil.copy(os.path.join(source_folder_image, image_file), target_folder_image)
        shutil.copy(os.path.join(source_folder_txt, txt_file), target_folder_txt)

copy_files(train_files, image_all_data_folder, txt_all_data_folder, 'image_train', 'txt_train')
copy_files(val_files, image_all_data_folder, txt_all_data_folder, 'image_val', 'txt_val')

print("ファイルのコピーが完了しました。")
