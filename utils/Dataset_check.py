'''
このコードは、YOLO形式のラベルをもとに画像にバウンディングボックスを描画し、
ラベル情報が正しく反映されているか確認するためのものです。
'''

import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_bbox(image_path, label_path, image_size):
    """
    画像にバウンディングボックスを描画する。

    :param image_path: 画像ファイルのパス
    :param label_path: 対応するラベルファイルのパス
    :param image_size: 画像のサイズ (幅, 高さ)
    """
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        # ラベルファイルを読み込み、各バウンディングボックスを描画
        with open(label_path, 'r') as file:
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.split())

                # YOLO形式を座標に変換
                x_center *= image_size[0]
                y_center *= image_size[1]
                width *= image_size[0]
                height *= image_size[1]
                left = x_center - width / 2
                top = y_center - height / 2
                right = x_center + width / 2
                bottom = y_center + height / 2

                draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # 画像を表示
        plt.imshow(img)
        plt.show()

# 使用例
image_directory = '/Users/nagasawa/Downloads/Label_Check'
image_size = (1440, 1440)

for file_name in sorted(os.listdir(image_directory))[:5]:
    if file_name.endswith('.png'):  # 画像ファイルのみ対象
        image_path = os.path.join(image_directory, file_name)
        label_path = os.path.join(image_directory, file_name.replace('.png', '.txt'))
        draw_bbox(image_path, label_path, image_size)
