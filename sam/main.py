# coding: utf-8
import argparse
import sys

from services.base_preprocess_service import PreprocessFactory
from services.sam_process_service import SamProcessService
from services.output_service import OutputService

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='数据集名称', default='MAR20', type=str)
    parser.add_argument('--use_gpu', help='是否使用 gpu', default=1, type=int)
    parser.add_argument('--parallel_num', help='多进程数量', default=1, type=int)
    args = parser.parse_args()
    
    preprocessor = PreprocessFactory().create(args.dataset)
    preprocess_result = preprocessor.call()

    processor = SamProcessService(ori_label_2_id_map=preprocessor.ori_label_2_id_map())
    process_result = processor.call(preprocess_result, use_gpu=args.use_gpu, parallel_num=args.parallel_num)

    output_service = OutputService(args.dataset)
    output_service.call(process_result, preprocessor.ori_label_2_id_map())
    
    sys.exit(0)
    

if __name__ == "__main__":
    main()