# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLOv5でオブジェクト検出を実行するスクリプト
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# ファイルパスやモデルの設定
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors, save_one_box

# モジュールインポート
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
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
    point_list=[],  # 中心座標を蓄積するリスト
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_csv=True,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # 保存先ディレクトリの設定
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # モデルをロード
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # データローダーを設定
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 推論の準備
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # 非最大抑制（NMS）を適用して結果を整理
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        csv_path = save_dir / "predictions.csv"
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        if 'accumulated_image' not in locals():
            accumulated_image = np.zeros((im0s[0].shape[0], im0s[0].shape[1], 3), dtype=np.uint8)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s[0].shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if save_csv:
                        write_to_csv(Path(path).name, names[int(cls)], conf.item())
                    center_x, center_y = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                    point_list.append((center_x, center_y))

        for point in point_list:
            cv2.circle(accumulated_image, point, radius=5, color=(0, 0, 255), thickness=-1)
        im0 = cv2.addWeighted(im0s[0], 1.0, accumulated_image, 0.5, 0)

        if view_img:
            cv2.imshow(str(path), im0)
            if cv2.waitKey(1) == ord('q'):
                break
        if save_img:
            cv2.imwrite(str(save_dir / f"{Path(path).stem}.jpg"), im0)

    LOGGER.info(f"Results saved to {save_dir}")
    if update:
        strip_optimizer(weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5s.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default="data/images", help="file/dir/URL/glob, 0 for webcam")
    parser.add_argument("--data", type=str, default="data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--img-size", "--img", "--imgsz", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results to CSV")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    opt = parser.parse_args()
    run(**vars(opt))
