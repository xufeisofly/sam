# coding: utf-8

from services.base_preprocess_service import PreprocessFactory
from services.sam_process_service import SamProcessService
from services.output_service import OutputService

if __name__ == '__main__':
    preprocessor = PreprocessFactory().create('MAR20')
    preprocess_result = preprocessor.call()

    processor = SamProcessService(ori_label_2_id_map=preprocessor.ori_label_2_id_map())
    process_result = processor.call(preprocess_result, use_gpu=False, parallel_num=1)

    output_service = OutputService('MAR20')
    output_service.call(process_result, preprocessor.ori_label_2_id_map())