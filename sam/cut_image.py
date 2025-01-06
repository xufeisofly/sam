# coding: utf-8
import argparse
import sys
import cv2
import glob
import os
import numpy as np
from PIL import Image

def get_file_name_without_ext(file_path: str) -> str:
    return file_path.split("/")[-1].split(".")[0]

def get_all_files(folder_path):
    # 使用递归匹配所有文件
    return glob.glob(os.path.join(folder_path, '**', '*'), recursive=True)


def cut_dataset(dataset_path):
    output_root = os.path.join(dataset_path, 'cut')
    os.makedirs(output_root, exist_ok=True)
    for dir in ['img_dir', 'ann_dir']:
        output_folder = os.path.join(output_root, dir)
        os.makedirs(output_folder, exist_ok=True)
        img_dir = os.path.join(dataset_path, dir)
        for sub_folder in os.listdir(img_dir):
            sub_output_folder = os.path.join(output_folder, sub_folder)
            os.makedirs(sub_output_folder, exist_ok=True)
            sub_folder_path = os.path.join(img_dir, sub_folder)
            all_files = get_all_files(sub_folder_path)
            for idx, file in enumerate(all_files):
                cut_image(file, sub_output_folder)
                print(f"==== 切片完成 {idx}/{len(all_files)}")
                
            
def slice_image(image, a, overlap=256):
    height, width, channels = image.shape
    slices = []

    # 计算切片的步长
    step = a - overlap

    # 切片的数量
    for i in range(0, height, step):
        for j in range(0, width, step):
            # 计算当前切片的结束位置
            end_i = min(i + a, height)
            end_j = min(j + a, width)
            
            # 如果当前切片不足大小，填充为 0
            slice_ = image[i:end_i, j:end_j]
            if slice_.shape[0] < a or slice_.shape[1] < a:
                # 创建一个形状为 (a, a, channels) 的零数组
                padded_slice = np.zeros((a, a, channels), dtype=image.dtype)
                # 将当前切片放入到零数组中
                padded_slice[:slice_.shape[0], :slice_.shape[1]] = slice_
                slices.append(padded_slice)
            else:
                slices.append(slice_)

    return slices


def save_slices_as_tif(slices, file_name, output_folder="output"):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, slice_ in enumerate(slices):
        # 将切片保存为 .tif 格式
        slice_image = Image.fromarray(slice_)
        # 构造文件名，例如：file_name_0.tif, file_name_1.tif, ...
        output_path = os.path.join(output_folder, f"{file_name}_{idx}.tif")
        slice_image.save(output_path)
        
    
def cut_image(file, output_folder):
    image = cv2.imread(file)
    width, height = image.shape[:2]
    # 获得边长
    a = 1024
    if width < 1024 or height < 1024:
        a = min(width, height)
        
    slices = slice_image(image, a)
    save_slices_as_tif(slices, get_file_name_without_ext(file), output_folder=output_folder)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='数据集路径', default='output/MAR20', type=str)
    args = parser.parse_args()
    
    cut_dataset(args.dataset_path)
    
    print(f"==== 完成 {args.dataset_path}")
    
    sys.exit(0)
    
    

if __name__ == "__main__":
    main()