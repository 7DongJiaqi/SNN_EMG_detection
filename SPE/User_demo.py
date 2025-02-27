from User_data_processing import *
import os
import re
import csv


SingleUserEvaluation('./preprocessed', 'mlp', 100, 50, 0.01)

with open('total_mean_values.txt', 'r') as file:
    lines = file.readlines()
    values = [float(line.strip()) for line in lines]
    overall_mean = sum(values) / len(values) if values else 0
print("spike rate:", overall_mean)
os.remove('total_mean_values.txt')

# with open('qsn3/aba_85_18_c9.txt', 'a') as file:
#     file.write(f"Spike rate: {overall_mean}\n")

# def read_and_extract_numbers(filename):
#     numbers = []
#     lines_to_read = set(range(3300, 3609, 11))  # 初始化包括3130和3161在内的行号集合
#     # 更新集合以包括3161之后每31行的行号
#     start_line = 3310
#     with open(filename, 'r') as file:
#         for i, line in enumerate(file, 1):  # 从1开始计数行号
#             if i >= start_line:
#                 if (i - start_line) % 11 == 0:
#                     lines_to_read.add(i)
#             if i in lines_to_read:
#                 # 使用正则表达式提取数字
#                 match = re.search(r"AVG_F1_Score:\s*(\d+\.\d+)", line)
#                 if match:
#                     # 将提取的数字（字符串形式）转换为浮点数并添加到列表
#                     # print(round(float(match.group(1)),4))
#                     numbers.append(round(float(match.group(1)),4))
#     return numbers

# # 假设数据文件名为"data.txt"
# filename= "qsn3/aba_85_18_c9.txt"
# numbers = read_and_extract_numbers(filename)

# def write_to_csv(original_filepath, numbers, output_csv_path):
#     filename = os.path.basename(original_filepath)
#     parts = filename.split("_")
#     n1 = int(parts[0][1:])  # 提取 `n1` 中的数字部分
#     num = int(parts[1].split(".")[0])  # 提取 `99` 部分，去掉扩展名

#     mean_value = sum(numbers) / len(numbers) if numbers else 0

#     with open(output_csv_path, mode='a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([n1, num, numbers, mean_value])

# original_file = filename
# output_csv = "output.csv"

# write_to_csv(original_file, numbers, output_csv)

# print(numbers)

# import statistics as st
# print(st.mean(numbers))