import os
import csv
import tqdm
csv_file_path = "E:\PythonProject\dualModalAttention\src\dataset\Celeb_label.csv"
real_img_file_path = "E:\PythonProject\dualModalAttention\src\dataset\Celeb-real\images"
synthesis_img_file_path = "E:\PythonProject\dualModalAttention\src\dataset\Celeb-synthesis\images"
with open(csv_file_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([11, 11])
    # for img in os.listdir(real_img_file_path):
    #     # 更改路径的分隔符
    #     name = img.split('.')[0]
    #     label = '1'
    #     writer.writerow([img, label])
    # for img in os.listdir(synthesis_img_file_path):
    #     # 更改路径的分隔符
    #     name = img.split('.')[0]
    #     label = '0'
    #     writer.writerow([img, label])
    print('written into csv file:', csv_file_path)