import json
import os
import shutil

def convert_labelbox_to_yolo(label_data, image_width, image_height):
    """
    LabelBox形式のアノテーションをYOLO形式に変換する。
    :param label_data: バウンディングボックス情報のリスト
    :param image_width: 画像の幅
    :param image_height: 画像の高さ
    :return: YOLO形式の文字列リスト
    """
    yolo_labels = []
    for obj in label_data:
        bbox = obj.get("bbox", {})
        x_center = bbox.get("left", 0) + bbox.get("width", 0) / 2
        y_center = bbox.get("top", 0) + bbox.get("height", 0) / 2

        # 座標を画像サイズで正規化
        x_center /= image_width
        y_center /= image_height
        width = bbox.get("width", 0) / image_width
        height = bbox.get("height", 0) / image_height

        yolo_labels.append(f"0 {x_center} {y_center} {width} {height}")

    return yolo_labels

def process_labelbox_json_v2(json_path, output_dir, image_size=(1440, 1440)):
    # JSONデータを読み込む
    with open(json_path, "r") as file:
        data = json.load(file)

    os.makedirs(output_dir, exist_ok=True)

    for item in data:
        label_data = item.get("Label", {}).get("objects", [])
        yolo_labels = convert_labelbox_to_yolo(label_data, *image_size)
        image_file_name = item.get("External ID", "").split(".")[0]
        label_file_name = f"{image_file_name}.txt"
        file_path = os.path.join(output_dir, label_file_name)
        with open(file_path, "w") as file:
            for label in yolo_labels:
                file.write(label + "\n")

    # 出力ディレクトリをZIP形式で圧縮
    zip_path = f"{output_dir}.zip"
    shutil.make_archive(output_dir, "zip", output_dir)
    return zip_path

# 使用例
json_file_path = '/Users/nagasawa/Downloads/UAS_No_Avoid_h2.json'
output_directory = '/Users/nagasawa/Downloads/GraduationThesis/New_Dataset/UAS_No_Avoid_h2'
zip_file_path = process_labelbox_json_v2(json_file_path, output_directory)
print(f"YOLOv5 labels are saved and zipped at: {zip_file_path}")

'''
出力形式についての説明：
<class_id> <x_center> <y_center> <width> <height> 形式で出力され、座標は正規化されます。
'''
