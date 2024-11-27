# coding: utf-8
import os
import numpy as np

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
from typing import List
from lxml import etree

class BridgePreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "BridgeDataset") if dataset_path is None else dataset_path

    def call(self, limit=-1) -> PreprocessResult:
        anno_folder = os.path.join(self._dataset_path, "bridges_dataset/Annotations")
        all_xml_files = os.listdir(anno_folder)
        all_xml_files = [f for f in all_xml_files if f.endswith('.xml')]

        result = PreprocessResult()
        for xml_file in all_xml_files:
            tree = etree.parse(os.path.join(anno_folder, xml_file))
            img_file_path = os.path.join(self._dataset_path, 'bridges_dataset', 'JPEGImages', os.path.splitext(xml_file)[0] + ".jpg")
            if not os.path.exists(img_file_path):
                print(img_file_path, "is missing! ")
                continue
            root = tree.getroot()
            objects = root.xpath("//object")
            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType.TRAIN)

            for obj in objects:
                box_array = np.array([int(obj.find('bndbox/xmin').text),
                                int(obj.find('bndbox/ymin').text),
                                int(obj.find('bndbox/xmax').text),
                                int(obj.find('bndbox/ymax').text)])

                result_item.append(BoxItem(
                    ori_label=self._get_ori_label_by_token(obj.find('name').text),
                    box_array=box_array))

            result.append(result_item)

        return result    

    def ori_label_2_id_map(self) -> dict:
        return {
            'bridge': 255
        }

    def _get_ori_label_by_token(self, label_token: str) -> str:
        token2ori = {
            'bridge': 'bridge'
        }
        return token2ori[label_token]


   