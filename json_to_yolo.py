import json  # JSONデータを扱うためのモジュールをインポート
import os  # OS操作のためのモジュールをインポート
import shutil  # ファイル操作のためのモジュールをインポート

def convert_labelbox_to_yolo(label_data, image_width, image_height):
    """
    LabelBox形式のアノテーションをYOLO形式に変換する関数

    :param label_data: LabelBoxからのバウンディングボックス情報のリスト
    :param image_width: 画像の幅
    :param image_height: 画像の高さ
    :return: YOLO形式の文字列リスト
    """
    yolo_labels = []  # YOLOフォーマットのラベルを格納するリスト
    for obj in label_data:
        # バウンディングボックスの座標を取得
        bbox = obj.get('bbox', {})
        x_center = bbox.get('left', 0) + bbox.get('width', 0) / 2  # 中心x座標を計算
        y_center = bbox.get('top', 0) + bbox.get('height', 0) / 2  # 中心y座標を計算

        # 座標を正規化
        x_center /= image_width  # x座標を画像幅で割り、0〜1の範囲に正規化
        y_center /= image_height  # y座標を画像高さで割り、0〜1の範囲に正規化
        width = bbox.get('width', 0) / image_width  # バウンディングボックスの幅を正規化
        height = bbox.get('height', 0) / image_height  # バウンディングボックスの高さを正規化

        class_id = 0  # クラスID（データセットに基づき設定）
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")  # YOLO形式でラベルを追加

    return yolo_labels  # YOLO形式のラベルリストを返す

def process_labelbox_json_v2(json_path, output_dir, image_size=(1440, 1440)):
    # JSONファイルを開いてデータを読み込む
    with open(json_path, 'r') as file:
        data = json.load(file)

    os.makedirs(output_dir, exist_ok=True)  # 出力ディレクトリが存在しない場合は作成

    for item in data:
        label_data = item.get('Label', {}).get('objects', [])  # バウンディングボックス情報を取得
        yolo_labels = convert_labelbox_to_yolo(label_data, *image_size)  # YOLO形式に変換
        image_file_name = item.get('External ID', '').split('.')[0]  # 画像ファイル名の基本部分を取得
        label_file_name = f"{image_file_name}.txt"  # ラベルファイルに同じ基本名を使用
        file_path = os.path.join(output_dir, label_file_name)  # 出力先のパスを作成
        with open(file_path, 'w') as file:
            for label in yolo_labels:
                file.write(label + "\n")  # ラベルを1行ずつ書き込む

    zip_path = f"{output_dir}.zip"  # 圧縮ファイルのパス
    shutil.make_archive(output_dir, 'zip', output_dir)  # 出力ディレクトリをZIPファイルに圧縮
    return zip_path  # 圧縮ファイルのパスを返す

# 実際の使用例
json_file_path = '/Users/nagasawa/Downloads/UAS_No_Avoid_h2.json'  # LabelBoxのJSONファイルへのパス
output_directory = '/Users/nagasawa/Downloads/GraduationThesis/New_Dataset/UAS_No_Avoid_h2'  # YOLOラベルファイルを保存するディレクトリ

# JSONファイルを修正された関数で処理
zip_file_path = process_labelbox_json_v2(json_file_path, output_directory)
print(f"YOLOv5 labels are saved and zipped at: {zip_file_path}")  # ZIPファイルの保存場所を表示


'''
zipで出力するわけ

jsonファイルには複数の画像のラベルデータが一つのファイルにまとめられている
出力するyoloファイルは、画像ごとにtxtファイルを出力する
そのため、出力はファイルがばーっと出てきてしまうので、フォルダとして出力させるためzipファイルにした

出力されるyoloファイル
<class_id> <x_center> <y_center> <width> <height>  → 0 0.0694 0.0833 0.0694 0.0833
x_center と y_center は、バウンディングボックスの中心。そして、バウンディングボックスの横と縦の長さでバウンディングボックスを表現。
この値は正規化（元の画像全体を 幅1、高さ1 ）
'''