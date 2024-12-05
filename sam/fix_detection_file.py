# coding: utf-8
import argparse
import sys
import logging
import json
import os
import traceback

from services.base_preprocess_service import PreprocessFactory
from services.sam_process_service import SamProcessService
from services.output_service import OutputService
import torch
from util.logger import init_logging, logger, loglevel, set_loglevel


def process_failure(dataset, e: Exception, offset, chunk, classify_dict, err_stack):
    file = dataset + "_fail.json"
    
    if os.path.exists(file) and os.path.getsize(file) > 0:
        if offset == 0:
            os.remove(file)# 检查文件是否存在且非空
        with open(file, 'r') as f:
            try:
                fail_info = json.load(f)  # 尝试读取 JSON 内容
            except json.JSONDecodeError:  # 如果文件内容损坏，初始化为空列表
                fail_info = []
    else:
        fail_info = []

    fail_info.append({
        "offset": offset,
        "chunk": chunk,
        "error": str(e),
        "stack": err_stack,
    })
    
    info = {
        "failures": fail_info,
        "classify_dict": classify_dict
    }
    
    with open(file, 'w') as f:
        json.dump(info, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='数据集名称', default='MAR20', type=str)
    parser.add_argument('--dataset_path', help='指定数据集路径，不设置的话使用默认', default=None, type=str)
    parser.add_argument('--parallel_num', help='多进程数量，默认与 GPU 核数相同', default=0, type=int)
    parser.add_argument('--limit', help='图片处理数量 for train, val, test，默认处理所有', default=-1, type=int)
    parser.add_argument('--merge_mask', help='是否合并 mask 文件', default=1, type=int)
    parser.add_argument('--chunk', help='分批处理, -1=nochunk', default=100, type=int)
    parser.add_argument('--low_memory', help='低内存模式，会将部分大变量通过硬盘读取 0-默认模式，1-低内存模式', default=0, type=int)
    parser.add_argument('--loglevel', type=str, default='INFO',
                    help='Set the log level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL')
    args = parser.parse_args()
    
    
    logger.debug(f"==== 开始处理 {args.dataset}")
    preprocessor = PreprocessFactory().create(args.dataset, dataset_path=args.dataset_path)
    preprocess_result = preprocessor.call(limit=args.limit)
    
    logger.info(f"==== 完成预处理 {args.dataset}")

    processor = SamProcessService(
        ori_label_2_id_map=preprocessor.ori_label_2_id_map(), 
        use_gpu=0, 
        parallel_num=1)
    
    output_service = OutputService(args.dataset)
    process_result_without_masks = processor.get_result_without_mask(preprocess_result, merge_mask=bool(args.merge_mask))
    output_service.fix_detection_data(process_result_without_masks, preprocessor.ori_label_2_id_map())
    
    logger.info(f"==== 完成 rest 数据保存 {args.dataset}")
    
    sys.exit(0)
    
    
def get_classify_dict(train_json_file_path, val_json_file_path, test_json_file_path):
    with open(train_json_file_path, 'r') as f:
        train_json = json.load(f)
    

if __name__ == "__main__":
    main()