import cv2  # OpenCVをインポートして動画や画像を扱う
import os  # OS操作のためのモジュールをインポート
from pathlib import Path  # フォルダ操作のためのモジュールをインポート

def extract_frames_improved(video_path, target_folder, num_frames=100):
    """
    動画から指定した枚数のフレームを均等に抽出し、フォルダに保存する関数。

    :param video_path: 動画ファイルのパス
    :param target_folder: フレームを保存するフォルダ
    :param num_frames: 抽出するフレーム数
    """
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)  # 保存先フォルダが存在しない場合は作成

        cap = cv2.VideoCapture(video_path)  # 動画ファイルを読み込む
        if not cap.isOpened():  # 動画が正常に読み込めない場合
            print(f"Error: Cannot open video file {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の総フレーム数を取得
        frame_interval = max(1, total_frames // num_frames)  # フレーム間隔を計算

        for i in range(num_frames):
            frame_index = i * frame_interval  # 抽出するフレームの位置を計算
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # フレーム位置を設定
            ret, frame = cap.read()  # 指定位置のフレームを読み込む
            if not ret:  # フレームが正常に読み込めなかった場合
                print(f"Error: Could not read frame at index {frame_index}")
                continue

            frame_file_path = os.path.join(target_folder, f"frame_{i:04d}.png")  # 保存ファイルパスを作成
            cv2.imwrite(frame_file_path, frame)  # 画像として保存
            print(f"Saved: {frame_file_path}")  # 保存完了メッセージを表示

    except Exception as e:
        print(f"An error occurred: {e}")  # エラーメッセージを表示
    finally:
        cap.release()  # 動画を閉じる
