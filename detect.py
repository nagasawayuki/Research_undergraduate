# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLOv5でオブジェクト検出を実行するスクリプト
"""

import argparse  # 引数解析のためのモジュール
import csv  # CSVファイルを扱うためのモジュール
import os  # OS操作のためのモジュール
import platform  # プラットフォームを確認するためのモジュール
import sys  # システム操作のためのモジュール
from pathlib import Path  # ファイルやディレクトリのパス操作のためのモジュール
import cv2  # OpenCVライブラリをインポート
import numpy as np  # 数値計算ライブラリ
import torch  # PyTorchライブラリ

FILE = Path(__file__).resolve()  # スクリプトの絶対パスを取得
ROOT = FILE.parents[0]  # YOLOv5のルートディレクトリを設定
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ルートディレクトリをシステムパスに追加
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # カレントディレクトリからの相対パスに変換

from ultralytics.utils.plotting import Annotator, colors, save_one_box  # プロット用の関数をインポート

# 必要なモジュールをインポート
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode  # デバイスの選択と推論モードの設定

@smart_inference_mode()
def run(
    point_list = [],  # 中心座標を蓄積するリスト　!!!!!!!!!!!!!!!!!!!!!!!!
    weights=ROOT / "yolov5s.pt",  # モデルのパス
    source=ROOT / "data/images",  # 入力ソース（画像、動画、ウェブカムなど）
    data=ROOT / "data/coco128.yaml",  # データセットのパス
    imgsz=(640, 640),  # 推論時の画像サイズ !!!!!!!!!!!!!!!!!!!!!!!!
    conf_thres=0.25,  # 信頼度の閾値
    iou_thres=0.45,  # NMSのIoU閾値
    max_det=1000,  # 1画像あたりの最大検出数
    device="",  # 使用デバイス（CPU/GPU）
    view_img=False,  # 結果を表示するか
    save_txt=False,  # テキスト形式で結果を保存するか
    save_csv=True,  # CSV形式で結果を保存するか
    save_conf=False,  # 信頼度をテキストに保存するか
    save_crop=False,  # 検出ボックスを切り取って保存するか
    nosave=False,  # 画像や動画を保存しない
    classes=None,  # クラスでフィルタリング
    agnostic_nms=False,  # クラスに依存しないNMSを使用
    augment=False,  # 拡張推論を使用するか
    visualize=False,  # 特徴を可視化するか
    update=False,  # モデルを更新するか
    project=ROOT / "runs/detect",  # 結果を保存するプロジェクトディレクトリ
    name="exp",  # 保存先の名前
    exist_ok=False,  # 名前の重複を許可
    line_thickness=3,  # バウンディングボックスの線の太さ
    hide_labels=False,  # ラベルを隠すか
    hide_conf=False,  # 信頼度を隠すか
    half=False,  # FP16精度を使用するか
    dnn=False,  # OpenCV DNNを使用するか
    vid_stride=1,  # ビデオフレームの間引き
):
    source = str(source)  # 入力ソースを文字列に変換
    save_img = not nosave and not source.endswith(".txt")  # 画像の保存を設定
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # ファイル形式の確認
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))  # URLかどうか確認
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)  # ウェブカムかどうか確認
    screenshot = source.lower().startswith("screen")  # スクリーンキャプチャかどうか確認
    if is_url and is_file:
        source = check_file(source)  # ソースがURLならローカルにダウンロード

    # 結果の保存先ディレクトリを設定
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 結果保存パスをインクリメント
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # ディレクトリを作成

    # モデルを読み込む
    device = select_device(device)  # デバイス選択（CPU/GPU）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # モデル読み込み
    stride, names, pt = model.stride, model.names, model.pt  # モデルのストライドと名前を取得
    imgsz = check_img_size(imgsz, s=stride)  # 画像サイズをストライドに合わせて調整

    # データローダーを設定
    bs = 1  # バッチサイズを1に設定
    if webcam:
        view_img = check_imshow(warn=True)  # ウェブカムビューが有効か確認
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # ウェブカムデータをロード
        bs = len(dataset)  # バッチサイズをデータセットサイズに設定
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)  # スクリーンショットのデータをロード
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 画像データをロード
    vid_path, vid_writer = [None] * bs, [None] * bs  # ビデオのパスとライターを初期化

    # 推論を実行
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # モデルをウォームアップ
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))  # プロファイルを設定

    for path, im, im0s, vid_cap, s in dataset:  # データセット内の各画像またはフレームをループ
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # 画像をデバイス上のテンソルに変換
            im = im.half() if model.fp16 else im.float()  # データ型を半精度または浮動小数点に変換
            im /= 255  # ピクセル値を0～1に正規化
            if len(im.shape) == 3:
                im = im[None]  # バッチ次元を追加
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)  # 画像をバッチ単位に分割

        # 推論
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # 可視化設定
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)  # 推論結果を蓄積
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]  # 推論結果をリストに格納
            else:
                pred = model(im, augment=augment, visualize=visualize)  # 推論を実行

        # NMSで結果をフィルタリング
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # NMSで重複除去

        # CSVファイルのパスを設定
        csv_path = save_dir / "predictions.csv"  # CSVファイル保存先を設定

        # CSVファイルへの書き込み関数を定義 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        def write_to_csv(image_name, prediction, confidence): #引数(画像名,オブジェクトクラス,信頼度)
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}  # 書き込みデータを準備
            with open(csv_path, mode="a", newline="") as f: #mode="a" は、ファイルを追記モードで開く
                writer = csv.DictWriter(f, fieldnames=data.keys())  # ("Image Name"、"Prediction"、"Confidence"）をCSVのヘッダー行として使用
                if not csv_path.is_file():
                    writer.writeheader()  # CSVのヘッダーを書き込む
                writer.writerow(data)  # データを書き込む
                
        # アキュムレーションバッファが初期化されていない場合に作成
        if 'accumulated_image' not in locals():
            accumulated_image = np.zeros((im0s[0].shape[0], im0s[0].shape[1], 3), dtype=np.uint8)  # バッファをゼロで初期化

        # 各検出ポイントをリストに蓄積 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # フレームまたは画像ごとの検出結果を順に処理
        for det in pred: #pred は model の推論結果 
            if len(det): #detの要素数が0なら次へ
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s[0].shape).round()  # 推論サイズから元画像に変換したときのバウンディングボックス座標を格納
                for *xyxy, conf, cls in reversed(det): #detには、更新されたバウンディングボックス座標とconf(信頼度),cls(オブジェクトクラス)
                    if save_csv:
                        write_to_csv(Path(path).name, names[int(cls)], conf.item())  # CSVに結果を保存
                    center_x, center_y = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)  # 中心座標を計算
                    point_list.append((center_x, center_y))  # 中心座標をリストに追加
                    
        # 蓄積した各ポイントを描画して表示
        for point in point_list:
            cv2.circle(accumulated_image, point, radius=5, color=(0, 0, 255), thickness=-1)  # 各中心点を塗りつぶし赤円で描画
        im0 = cv2.addWeighted(im0s[0], 1.0, accumulated_image, 0.5, 0)  # 重ね合わせて表示。赤点を半透明に

        # 出力処理がある場合、フレームの表示または保存を実行
        if view_img:
            cv2.imshow(str(path), im0)  # 結果をウィンドウに表示
            if cv2.waitKey(1) == ord('q'):
                break  # 'q'キーで停止
        if save_img:
            cv2.imwrite(str(save_dir / f"{Path(path).stem}.jpg"), im0)  # 結果画像を保存
    LOGGER.info(f"Results saved to {save_dir}")  # 結果保存のログを出力
    if update:
        strip_optimizer(weights)  # モデルの最適化を解除

if __name__ == "__main__":
    # メイン処理: パラメータを引数から取得し、run関数を実行
    parser = argparse.ArgumentParser()  # 引数パーサーを作成
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5s.pt", help="model path(s)")  # モデルパス引数
    parser.add_argument("--source", type=str, default="data/images", help="file/dir/URL/glob, 0 for webcam")  # 入力ソース
    parser.add_argument("--data", type=str, default="data/coco128.yaml", help="(optional) dataset.yaml path")  # データセット
    parser.add_argument("--img-size", "--img", "--imgsz", nargs="+", type=int, default=[640], help="inference size h,w")  # 画像サイズ
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")  # 信頼度閾値
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")  # NMS IoU閾値
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")  # 最大検出数
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  # デバイス
    parser.add_argument("--view-img", action="store_true", help="display results")  # 表示オプション
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")  # テキスト保存オプション
    parser.add_argument("--save-csv", action="store_true", help="save results to CSV")  # CSV保存オプション
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")  # プロジェクトパス
    parser.add_argument("--name", default="exp", help="save results to project/name")  # 結果保存先名
    opt = parser.parse_args()  # 引数を解析して変数に格納
    run(**vars(opt))  # 解析した引数でrun関数を実行
