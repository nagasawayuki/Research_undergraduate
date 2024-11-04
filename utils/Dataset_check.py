'''
このコードは、画像に対応するYOLO形式のラベルデータをもとにバウンディングボックスを描画し、
ラベル情報が正しく反映されているか確認するためのものです。
json_to_yolo.pyの確認用ファイル
'''

import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_bbox(image_path, label_path, image_size):
    """
    Draw bounding boxes on the image.

    :param image_path: Path to the image file.
    :param label_path: Path to the corresponding label file.
    :param image_size: Size of the image (width, height).
    """
    # Load the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        # Read label file and draw each bbox
        with open(label_path, 'r') as file:
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.split())

                # Convert YOLO format to bbox coordinates
                x_center *= image_size[0]
                y_center *= image_size[1]
                width *= image_size[0]
                height *= image_size[1]
                left = x_center - width / 2
                top = y_center - height / 2
                right = x_center + width / 2
                bottom = y_center + height / 2

                # Draw bbox
                draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Display the image
        plt.imshow(img)
        plt.show()

# Update these paths and values
image_directory = '/Users/nagasawa/Downloads/Label_Check'  # Directory containing images and labels
image_size = (1440, 1440)  # Size of the images

# Processing the first few images
for file_name in sorted(os.listdir(image_directory))[:5]:
    if file_name.endswith('.png'):  # or '.jpg', depending on your image format
        image_path = os.path.join(image_directory, file_name)
        label_path = os.path.join(image_directory, file_name.replace('.png', '.txt'))  # Update extension if needed
        draw_bbox(image_path, label_path, image_size)
