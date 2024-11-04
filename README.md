# 全方位カメラでのドローン検出用YOLOモデル

本プロジェクトは、全方位カメラを使用してドローンを検出するYOLOモデルを作成することを目的としています。以下は、この目標を達成するためのワークフローの概要です。

## 研究フロー

### 1. データセット作成
- **ビデオ撮影**: 衝突するケースと衝突しないケースを各方向で網羅した動画を撮影（図を含む）。
- **画像抽出**: 各動画から100枚の画像を抽出。[`src/movie_to_image.py`](#src_movie_to_image_py)
- **アノテーション**: [`LabelBox`](https://labelbox.com/)を使用して画像にアノテーション。
- **形式変換**: JSONアノテーションファイルをYOLO形式のTXTファイルに変換。[`src/json_to_yolo.py`](#src_json_to_yolo_py)
- **データ分割**: 画像とアノテーションセットをランダムに80％をトレーニング用、20％をテスト用に分割し、それぞれ`train`および`vali`フォルダに配置。[`src/mv_make_train&val_from_all.py`](#src_mv_make_train_val_from_all_py)

### 2. モデル作成
- **YOLOv5トレーニング**: 準備したデータセットを使用してYOLOv5フレームワークでモデルを作成（実行コマンドを添付）。
- **モデル評価**: 作成したモデルを評価し、評価メトリクスを示す画像を添付。

### 3. 物体検出
- **出力カスタマイズ**: [`src/detect.py`](#src_detect_py)
  - 検出位置の軌跡を可視化するように出力を変更。
  - 検出位置をCSVファイルにエクスポート。
- **テスト**: テストデータを使用して検出。

### 4. 検出結果を使った考察
- **CSV分析**: CSVファイルをExcelに読み込み、さらなる分析を実施。
- **特徴分析**: 微分値（フレームごとの差）を使用して特徴を分析。
- **移動平均**: 移動平均を計算。
- **プロット**: 操作した値をプロット。[`src/csv_plot.py`](#src_csv_plot_py)

---

## ファイルと機能

### <a id="src_detect_py">**`src/detect.py`**</a>
YOLOモデルを使用して物体検出を行い、結果をカスタマイズするためのスクリプトです。検出した物体の位置をCSVに保存したり、可視化したりする機能があります。

### <a id="src_movie_to_image_py">**`src/movie_to_image.py`**</a>
動画ファイルからフレームを抽出し、静止画像として保存するためのスクリプトです。各動画から指定された枚数の画像を抽出し、データセット作成のために使用します。

### <a id="src_json_to_yolo_py">**`src/json_to_yolo.py`**</a>
LabelBoxなどのツールで生成されたJSON形式のアノテーションファイルを、YOLO形式のTXTファイルに変換するスクリプトです。これにより、YOLOモデルでトレーニング可能な形式にデータを整えます。出力されるyoloファイル
<class_id> <x_center> <y_center> <width> <height>  → 0 0.0694 0.0833 0.0694 0.0833
x_center と y_center は、バウンディングボックスの中心。そして、バウンディングボックスの横と縦の長さでバウンディングボックスを表現。
この値は正規化（元の画像全体を 幅1、高さ1 ）

### <a id="src_mv_make_train_val_from_all_py">**`src/mv_make_train&val_from_all.py`**</a>
データセット全体からランダムに画像とアノテーションを選択し、トレーニング用（train）と検証用（vali）に分割するスクリプトです。このスクリプトにより、データを適切に分けてモデルのトレーニングと評価が行えるようにします。

### <a id="src_csv_plot_py">**`src/csv_plot.py`**</a>
物体検出の結果を保存したCSVファイルを読み込み、データをプロットするスクリプトです。検出位置の変化や特徴を可視化するために使用します。








