# coding: utf-8
import os
import json
import numpy as np
from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
from util.box import poly2box


class AirplanePreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "Airplane") if dataset_path is None else dataset_path
    
    def _get_data_paths(self, data_type: str):
        image_dir = {
            "train": "train",
            "val":"val"
        }[data_type]

        json_dir = {
            "train": "train",
            "val":"val"
        }[data_type]

        image_path = os.path.join(self._dataset_path, image_dir)
        json_path = os.path.join(self._dataset_path, json_dir)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image directory {image_path} does not exist.")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation directory {json_path} does not exist.")

        return image_path, json_path


    def process_data_type(self, data_type: str, limit=-1) -> PreprocessResult:
        image_path, json_path = self._get_data_paths(data_type)

        all_json_files = os.listdir(json_path)
        all_json_files = [f for f in all_json_files if f.endswith('.json')]

        result = PreprocessResult()

        for json_file in all_json_files[:limit] if limit > 0 else all_json_files:
            with open(os.path.join(json_path, json_file), 'r') as file:
                json_data = json.load(file)

            img_file_name = os.path.splitext(json_file)[0] + ".tif"
            img_file_path = os.path.join(image_path, img_file_name)
            if not os.path.exists(img_file_path):
                print(f"Warning: Image file {img_file_path} not found.")
                continue

            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType[data_type.upper()])

            for feature in json_data['shapes']:
                pixel_coords = feature['points']
                box_array = poly2box(np.array([int(num) for num in pixel_coords.split(",")]))
                box_item = BoxItem(
                    ori_label=self._get_ori_label_by_token(feature['properties']['type_id']),
                    box_array=np.array(box_array)
                )
                result_item.append(box_item)    

            result.append(result_item)
        
        return result


    def call(self, limit=-1) -> dict:
        train_result = self.process_data_type("train",limit)
        val_result = self.process_data_type("val", limit)
        
        result = PreprocessResult(
            train_result_list=train_result.result_list,
            val_result_list=val_result.result_list)
        return result
        


    def ori_label_2_id_map(self) -> dict:
        return {
           "A" : 255,
           "B" : 254,
           "C" : 253,
           "D" : 252,
           "E" : 251,
           "F" : 250,
           "G" : 249,
           "H" : 248,
           "I" : 247,
           "J" : 246,
           "K" : 245 
        }

    def _get_ori_label_by_token(self, label_token: str) -> str:
        token2ori = {
           "A" : "A",
           "B" : "B",
           "C" : "C",
           "D" : "D",
           "E" : "E",
           "F" : "F",
           "G" : "G",
           "H" : "H",
           "I" : "I",
           "J" : "J",
           "K" : "K"
        }
        return token2ori[label_token]