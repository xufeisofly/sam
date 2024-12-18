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
    parser.add_argument('--use_gpu', help='是否使用 gpu', default=1, type=int)
    parser.add_argument('--gpu_ids', help='指定 gpu id eg 0,1,2,3 使用 list-gpu-ids 命令获取全部 gpu ids，仅 use_gpu=1 时生效', default='', type=str)
    parser.add_argument('--parallel_num', help='多进程数量，默认与 GPU 核数相同', default=0, type=int)
    parser.add_argument('--limit', help='图片处理数量 for train, val, test，默认处理所有', default=-1, type=int)
    parser.add_argument('--merge_mask', help='是否合并 mask 文件', default=1, type=int)
    parser.add_argument('--chunk', help='分批处理, -1=nochunk', default=100, type=int)
    parser.add_argument('--low_memory', help='低内存模式，会将部分大变量通过硬盘读取 0-默认模式，1-低内存模式', default=0, type=int)
    parser.add_argument('--loglevel', type=str, default='INFO',
                    help='Set the log level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL')
    args = parser.parse_args()
    
    log_level = getattr(logging, args.loglevel.upper(), logging.DEBUG)

    # 设置 logger
    set_loglevel(log_level)
    init_logging()
    
    logger.debug(f"==== 开始处理 {args.dataset}")
    preprocessor = PreprocessFactory().create(args.dataset, dataset_path=args.dataset_path)
    preprocess_result = preprocessor.call(limit=args.limit)
    
    logger.info(f"==== 完成预处理 {args.dataset}")

    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')] if args.gpu_ids else None
    processor = SamProcessService(
        ori_label_2_id_map=preprocessor.ori_label_2_id_map(), 
        use_gpu=args.use_gpu, 
        parallel_num=args.parallel_num, 
        gpu_ids=gpu_ids,
        low_memory=bool(args.low_memory))
    output_service = OutputService(args.dataset)
    output_service.clear_output()
    process_result_without_masks = processor.get_result_without_mask(preprocess_result, merge_mask=bool(args.merge_mask))
    
    classified_process_result, classify_dict = output_service.classify_result(process_result_without_masks)
    output_service.save_rest(classified_process_result, preprocessor.ori_label_2_id_map())
    
    logger.info(f"==== 完成 rest 数据保存 {args.dataset}")

    total = len(preprocess_result)
    chunk = args.chunk if args.chunk > 0 else total
    offset = 0
    
    while offset < total:
        try:
            process_result = processor.get_result(
                preprocess_result, 
                merge_mask=bool(args.merge_mask), 
                limit=chunk, offset=offset)
            
            output_service.save_masks(
                output_service.classify_result_by_dict(process_result, classify_dict),
                preprocessor.ori_label_2_id_map())
            
            logger.info(f"==== 处理并保存 Masks 成功 {offset}/{total}->{offset+chunk}/{total} {args.dataset}")
            offset += chunk
        except BaseException as e:            
            logger.error(f"==== 处理并保存 Masks 失败 {offset}/{total}->{offset+chunk}/{total} {args.dataset}")
            process_failure(args.dataset, e, offset, chunk, classify_dict, traceback.format_exc())
            offset += chunk
            continue
            
        
    logger.info(f"==== Masks 处理完毕 {args.dataset}") 
    
    sys.exit(0)
    
    
def list_gpu_ids():
    print(','.join([str(i) for i in range(torch.cuda.device_count())]))
    

if __name__ == "__main__":
    main()