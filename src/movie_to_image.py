import cv2
import os
from pathlib import Path

def extract_frames(video_path, target_folder, num_frames):
    """
    動画から指定された数のフレームを抽出し保存する関数。
    :param video_path: 動画ファイルのパス
    :param target_folder: フレームを保存するフォルダ
    :param num_frames: 抽出するフレーム数
    """
    try:
        # 保存先フォルダが存在しなければ作成
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return

        # 総フレーム数を取得し、フレーム間隔を計算
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_frames)

        for i in range(num_frames):
            frame_index = i * frame_interval  # 抽出位置を設定
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame at index {frame_index}")
                continue

            # 画像をファイルとして保存
            frame_file_path = os.path.join(target_folder, f"frame_{i:04d}.png")
            cv2.imwrite(frame_file_path, frame)
            print(f"Saved: {frame_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
