from User_data_processing_attack import *

SingleUserEvaluation('../data/Study1_medfilt11_EMG','mlp', 100, 50, 0.01, loadpath=None, savepath='model/mlp_3l')

with open('total_mean_values.txt', 'r') as file:
    lines = file.readlines()
    values = [float(line.strip()) for line in lines]
    overall_mean = sum(values) / len(values) if values else 0
print("spike rate:", overall_mean)
os.remove('total_mean_values.txt')

with open('snn/n3_42_hb.txt', 'a') as file:
    file.write(f"Spike rate: {overall_mean}\n")


import re
import os
import csv
def read_and_extract_numbers(filename):
    numbers = []
    lines_to_read = set(range(1100,1209, 11))
    start_line = 1110
    with open(filename, 'r') as file:
        for i, line in enumerate(file, 1):
            if i >= start_line:
                if (i - start_line) % 11 == 0:
                    lines_to_read.add(i)
            if i in lines_to_read:
                # 使用正则表达式提取数字
                match = re.search(r"AVG_F1_Score:\s*(\d+\.\d+)", line)
                if match:
                    numbers.append(round(float(match.group(1)),4))
    return numbers

filename = "snn/n3_42_hb.txt"
numbers = read_and_extract_numbers(filename)

def write_to_csv(original_filepath, numbers, output_csv_path):
    filename = os.path.basename(original_filepath)
    parts = filename.split("_")
    n1 = int(parts[0][1:])  # 提取 `n1` 中的数字部分
    num = int(parts[1].split(".")[0])  # 提取 `99` 部分，去掉扩展名

    mean_value = sum(numbers) / len(numbers) if numbers else 0

    with open(output_csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([n1, num, numbers, mean_value])

original_file = filename
output_csv = "output_hb.csv"

write_to_csv(original_file, numbers, output_csv)
# print(f"Data written to {output_csv}")
print(numbers)

import statistics as st
print(st.mean(numbers))