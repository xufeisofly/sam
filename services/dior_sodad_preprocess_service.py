# coding: utf-8
import os
import numpy as np
import json

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
from typing import List
from util.logger import logger
from util.box import coco2box


class DiorSodaDPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "SODA-D") if dataset_path is None else dataset_path
        self._image_path = os.path.join(self._dataset_path, "Images/Images/Images")
        self._anno_path = os.path.join(self._dataset_path, "Annotations")

    def call(self, limit=-1) -> PreprocessResult:
        train_anno_file = os.path.join(self._anno_path, 'train.json')
        val_anno_file = os.path.join(self._anno_path, 'val.json')
        test_anno_file = os.path.join(self._anno_path, 'test.json')
        
        train_result = self._get_result_from_anno_file(train_anno_file, DataType.TRAIN, limit)
        val_result = self._get_result_from_anno_file(val_anno_file, DataType.VAL, limit)
        test_result = self._get_result_from_anno_file(test_anno_file, DataType.TEST, limit)
        
        result = PreprocessResult(
            train_result_list=train_result.result_list,
            val_result_list=val_result.result_list,
            test_result_list=test_result.result_list)
        return result
        
        
    def _get_result_from_anno_file(self, anno_file_path: str, data_type: DataType, limit=-1) -> PreprocessResult:
        with open(anno_file_path, 'r') as file:
            anno_dict = json.load(file)

        grouped_box_data = self._group_box_data_by_image_id(anno_dict)
        
        result = PreprocessResult()
        
        counter = 0
        for image_info in anno_dict["images"]:
            file_name = image_info['file_name']
            img_file_path = os.path.join(self._image_path, file_name)
            if not self._has_file(img_file_path):
                logger.error(f"{file_name} is not exist")
                continue
            image_id = image_info['id']
            if image_id not in grouped_box_data:
                logger.warning(f'{image_id} has no bounding boxes')
                continue
            
            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=data_type)
            for box_info in grouped_box_data[image_id]:
                box_array = np.array(coco2box(box_info['bbox']))
                
                result_item.append(BoxItem(
                    ori_label=self._get_ori_label_by_token(box_info['category_id']),
                    box_array=box_array))

            result.append(result_item)
            counter += 1
            # 支持固定数量
            if limit > 0 and counter >= limit:
                break
            
        return result

        
    def _has_file(self, file_path: str) -> bool:
        return os.path.exists(file_path)
        
    def _group_box_data_by_image_id(self, anno_dict: dict) -> dict:
        grouped_box_data = {}
        for box_info in anno_dict['annotations']:
            key = box_info['image_id']
            if key not in grouped_box_data:
                grouped_box_data[key] = []
            grouped_box_data[key].append(box_info)
        return grouped_box_data
            

    def ori_label_2_id_map(self) -> dict:
        return {
            'person': 255,
            'rider': 254,
            'bicycle': 253,
            'motor': 252,
            'vehicle': 251,
            'traffic-sign': 250,
            'traffic-light': 249,
            'traffic-camera': 248,
            'warning-cone': 247,
            'ignore': 246
        }

    def _get_ori_label_by_token(self, label_token: int) -> str:
        token2ori = {
            1: 'person',
            2: 'rider',
            3: 'bicycle',
            4: 'motor',
            5: 'vehicle',
            6: 'traffic-sign',
            7: 'traffic-light',
            8: 'traffic-camera',
            9: 'warning-cone',
            10: 'ignore'
        }
        return token2ori[label_token]


   