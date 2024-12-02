# coding: utf-8
import os
import numpy as np
import json

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
from typing import List
from util.logger import logger
from util.box import coco2box, poly2box


class DiorSodaAPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "SODA-A") if dataset_path is None else dataset_path
        self._image_path = os.path.join(self._dataset_path, "Images")
        self._anno_path = os.path.join(self._dataset_path, "Annotations")

    def call(self, limit=-1) -> PreprocessResult:
        train_anno_folder = os.path.join(self._anno_path, 'train')
        val_anno_folder = os.path.join(self._anno_path, 'val')
        test_anno_folder = os.path.join(self._anno_path, 'test')
        
        train_result = self._get_result_from_anno_folder(train_anno_folder, DataType.TRAIN, limit)
        val_result = self._get_result_from_anno_folder(val_anno_folder, DataType.VAL, limit)
        test_result = self._get_result_from_anno_folder(test_anno_folder, DataType.TEST, limit)
        
        result = PreprocessResult(
            train_result_list=train_result.result_list,
            val_result_list=val_result.result_list,
            test_result_list=test_result.result_list)
        return result
        
        
    def _get_result_from_anno_folder(self, anno_file_folder: str, data_type: DataType, limit=-1) -> PreprocessResult:
        all_json_files = os.listdir(anno_file_folder)
        all_json_files = [f for f in all_json_files if f.endswith('.json') and not f.startswith('._')]       
        
        result = PreprocessResult()
        counter = 0
        for anno_file in all_json_files[:2]:
            anno_file_path = os.path.join(anno_file_folder, anno_file)
            with open(anno_file_path, 'r') as file:
                anno_dict = json.load(file)
                
            categories = anno_dict["categories"]
            category_map = {c['id']: c['name'] for c in categories}
            
            img_file_path = os.path.join(self._image_path, anno_dict["images"]["file_name"])
            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=data_type)
            for anno in anno_dict["annotations"]:
                poly = anno['poly']
                if len(poly) < 8:
                    logger.warning(f"skipping invalid polygon: {poly}")
                    continue
                box_array = poly2box(poly)
                
                result_item.append(BoxItem(
                    ori_label=category_map[anno['category_id']],
                    box_array=box_array))
                
            result.append(result_item)
            counter += 1
            if limit > 0 and counter >= limit:
                break
            
        return result
            

    def ori_label_2_id_map(self) -> dict:
        return {
            'airplane': 255,
            'helicopter': 254,
            'small-vehicle': 253,
            'large-vehicle': 252,
            'ship': 251,
            'container': 250,
            'storage-tank': 249,
            'swimming-pool': 248,
            'windmill': 247,
            'ignore': 246
        }


   