# coding: utf-8

from services.base_preprocess_service import PreprocessFactory
from services.sam_process_service import SamProcessService
from services.output_service import OutputService

if __name__ == '__main__':
    preprocessor = PreprocessFactory().create('MAR20')
    preprocess_result = preprocessor.call()

    processor = SamProcessService(ori_label_2_id_fn=preprocessor.ori_label_2_id)
    process_result = processor.call(preprocess_result)

    output_service = OutputService('MAR20')
    output_service.clear_all_images()
    output_service.save_to_ann_dir(
        *output_service.seperate_train_and_val_result(process_result))