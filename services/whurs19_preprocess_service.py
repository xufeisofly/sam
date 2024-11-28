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
from PIL import Image


class Whurs19PreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "WHU-RS19") if dataset_path is None else dataset_path

    def call(self, limit=-1) -> PreprocessResult:
        folder_names = os.listdir(self._dataset_path)
        result = PreprocessResult()
        
        for folder_name in folder_names:
            if folder_name.startswith("."):
                continue
            
            folder_path = os.path.join(self._dataset_path, folder_name)
            img_files = os.listdir(folder_path)
            
            counter = 0
            for img_file in img_files:
                img_file_path = os.path.join(folder_path, img_file)
                result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType.TRAIN)
                
                image = Image.open(img_file_path)
                box_array = np.array([0, 0, image.width, image.height])
                
                result_item.append(BoxItem(
                    ori_label=folder_name,
                    box_array=box_array))
                result.append(result_item)
                counter += 1
                if limit > 0 and counter >= limit:
                    break
                
        return result
            

    def ori_label_2_id_map(self) -> dict:
        return {
            'Airport': 255,
            'Beach': 254,
            'Bridge': 253,
            'Commercial': 252,
            'Desert': 251,
            'Farmland': 250,
            'footballField': 249,
            'Forest': 248,
            'Industrial': 247,
            'Meadow': 246,
            'Mountains': 245,
            'Park': 244,
            'Parking': 243,
            'Pond': 242,
            'Port': 241,
            'railwayStation': 240,
            'Residential': 239,
            'River': 238,
            'Viaduct': 237,
        }

   