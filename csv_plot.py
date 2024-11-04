import pandas as pd  # データ操作のためにpandasをインポート
import matplotlib.pyplot as plt  # プロットのためにmatplotlibをインポート
import numpy as np  # 数値計算のためにnumpyをインポート
import os  # ディレクトリ操作のためにosをインポート

# Excelファイルのパス
excel_file_path = '/Users/nagasawa/Downloads/2024_角変異_result.xlsx'

# グラフを保存する出力フォルダのパス
output_folder_path = '/Users/nagasawa/Downloads/GraduationThesis/New_Graphes'
os.makedirs(output_folder_path, exist_ok=True)  # 出力フォルダが存在しない場合は作成

# Excelファイルを読み込む
xl = pd.ExcelFile(excel_file_path)
sheet_count = len(xl.sheet_names)  # Excelファイル内のシート数を取得
print(f'Total sheets to process: {sheet_count}')  # シートの総数を表示

# 各シート（ワークブック）をループ処理
for index, sheet_name in enumerate(xl.sheet_names, start=1):
    print(f'Processing sheet {index} of {sheet_count}: {sheet_name}')  # 処理中のシートを表示
    # シートをDataFrameとして読み込む（ヘッダーなしを想定）
    df = xl.parse(sheet_name, header=None)

    # DataFrameが10列以上（11列目が存在する）かチェック
    if df.shape[1] > 10:
        k_data = df.iloc[:, 10].dropna()  # 11列目を抽出し、NaN値を削除
        k_data = k_data.replace([np.inf, -np.inf], np.nan).dropna()  # 無限値をNaNに置換して削除

        # 'K'列のデータをプロット
        plt.figure(figsize=(10, 6))  # グラフのサイズを設定
        plt.plot(k_data, label=f'{sheet_name} K values', color='blue')  # K値をラベルと色付きでプロット

        # グラフのタイトル、軸ラベル、範囲設定
        plt.title(f'{sheet_name} K Value Plot')  # プロットのタイトル
        plt.xlabel('Index')  # x軸のラベル
        plt.ylabel('K Value')  # y軸のラベル
        plt.legend()  # 凡例を表示

        # シート名を含むファイル名で出力フォルダにグラフを保存
        output_file_path = os.path.join(output_folder_path, f'{sheet_name}_K_values.png')
        plt.savefig(output_file_path)  # グラフをPNGファイルとして保存
        plt.close()  # メモリ解放のためにプロットを閉じる
