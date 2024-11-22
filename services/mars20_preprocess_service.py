# coding: utf-8
import os
import numpy as np

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR
from typing import List
from lxml import etree

class Mars20PreprocessService(BasePreprocessService):
    def __init__(self) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "MARS20")

    def call(self) -> PreprocessResult:
        anno_folder = os.path.join(self._dataset_path, "Annotations/Horizontal Bounding Boxes")
        all_xml_files = os.listdir(anno_folder)
        all_xml_files = [f for f in all_xml_files if f.endswith('.xml')]

        result = PreprocessResult([])
        for xml_file in all_xml_files:
            tree = etree.parse(os.path.join(anno_folder, xml_file))
            img_file_path = os.path.join(self._dataset_path, 'JPEGImages', 'JPEGImages', os.path.splitext(xml_file)[0] + ".jpg")
            root = tree.getroot()
            objects = root.xpath("//object")
            result_item = PreprocessResultItem(img_file_path=img_file_path)

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

    def ori_label_2_id_map(self) -> int:
        return {
            'SU-35': 255,
            'C-130': 254,
            'C-17': 253,
            'C-5': 252,
            'F-16': 251,
            'TU-160': 250,
            'E-3': 249,
            'B-52': 248,
            'P-3C': 247,
            'B-1B': 246,
            'E-8': 245,
            'TU-22': 243,
            'F-15': 242,
            'KC-135': 241,
            'F-22': 240,
            'FA-18': 239,
            'TU-95': 238,
            'KC-10': 237,
            'SU-34': 236,
            'SU-24': 235,
        }
    

    def _get_ori_label_by_token(self, label_token: str) -> str:
        token2ori = {
            'A1': 'SU-35',
            'A2': 'C-130',
            'A3': 'C-17',
            'A4': 'C-5',
            'A5': 'F-16',
            'A6': 'TU-160',
            'A7': 'E-3',
            'A8': 'B-52',
            'A9': 'P-3C',
            'A10': 'B-1B',
            'A11': 'E-8',
            'A12': 'TU-22',
            'A13': 'F-15',
            'A14': 'KC-135',
            'A15': 'F-22',
            'A16': 'FA-18',
            'A17': 'TU-95',
            'A18': 'KC-10',
            'A19': 'SU-34',
            'A20': 'SU-24'
        }
        return token2ori[label_token]


   