# coding: utf-8
import argparse
import sys
import logging

from services.base_preprocess_service import PreprocessFactory
from services.sam_process_service import SamProcessService
from services.output_service import OutputService
from util.logger import setup_logger, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='数据集名称', default='MAR20', type=str)
    parser.add_argument('--dataset_path', help='指定数据集路径，不设置的话使用默认', default=None, type=str)
    parser.add_argument('--use_gpu', help='是否使用 gpu', default=1, type=int)
    parser.add_argument('--parallel_num', help='多进程数量', default=0, type=int)
    parser.add_argument('--limit', help='图片处理数量 for train, val, test，默认处理所有', default=-1, type=int)
    parser.add_argument('--merge_mask', help='是否合并 mask 文件', default=1, type=int)
    parser.add_argument('--log-level', type=str, default='DEBUG',
                    help='Set the log level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL')
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper(), logging.DEBUG)

    # 设置 logger
    setup_logger(log_level)
    
    logger.info(f"==== 开始处理 {args.dataset}")
    preprocessor = PreprocessFactory().create(args.dataset, dataset_path=args.dataset_path)
    preprocess_result = preprocessor.call(limit=args.limit)
    
    logger.info(f"==== 完成预处理 {args.dataset}")

    processor = SamProcessService(ori_label_2_id_map=preprocessor.ori_label_2_id_map())
    process_result = processor.call(preprocess_result, use_gpu=args.use_gpu, parallel_num=args.parallel_num,
                                    merge_mask=bool(args.merge_mask))

    logger.info(f"==== 完成 SAM 处理 {args.dataset}")

    output_service = OutputService(args.dataset)
    output_service.call(process_result, preprocessor.ori_label_2_id_map())
    
    logger.info(f"==== 全部完成 {args.dataset}")
    
    sys.exit(0)
    

if __name__ == "__main__":
    main()