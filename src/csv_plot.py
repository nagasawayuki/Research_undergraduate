import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Excelファイルと出力フォルダのパス
excel_file_path = '/Users/nagasawa/Downloads/2024_角変異_result.xlsx'
output_folder_path = '/Users/nagasawa/Downloads/GraduationThesis/New_Graphes'
os.makedirs(output_folder_path, exist_ok=True)

# Excelファイルを読み込む
xl = pd.ExcelFile(excel_file_path)
sheet_count = len(xl.sheet_names)
print(f'Total sheets to process: {sheet_count}')

for index, sheet_name in enumerate(xl.sheet_names, start=1):
    print(f'Processing sheet {index} of {sheet_count}: {sheet_name}')
    df = xl.parse(sheet_name, header=None)

    # 必要なデータ列が存在するかチェック
    if df.shape[1] > 10:
        k_data = df.iloc[:, 10].dropna()  # 11列目を抽出
        k_data = k_data.replace([np.inf, -np.inf], np.nan).dropna()  # 無限値をNaNに置換して削除

        # プロット作成
        plt.figure(figsize=(10, 6))
        plt.plot(k_data, label=f'{sheet_name} K values')
        plt.title(f'{sheet_name} K Value Plot')
        plt.xlabel('Index')
        plt.ylabel('K Value')
        plt.legend()

        # グラフを保存
        output_file_path = os.path.join(output_folder_path, f'{sheet_name}_K_values.png')
        plt.savefig(output_file_path)
        plt.close()

