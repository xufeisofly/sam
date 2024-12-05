# coding: utf-8
import os
import numpy as np

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
from typing import List
from lxml import etree

class HRRSDPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join("/root/liuh/sam/origin_data/", "HRRSD/TGRS-HRRSD-Dataset/OPT2017") if dataset_path is None else dataset_path

    def call(self, limit=-1) -> PreprocessResult:
        anno_folder = os.path.join(self._dataset_path, "Annotations/")
        all_xml_files = os.listdir(anno_folder)
        all_xml_files = [f for f in all_xml_files if f.endswith('.xml') and not f.startswith('._')]
        all_xml_files.sort()

        result = PreprocessResult()
        i = 0
        for xml_file in all_xml_files:
            # if limit == 0:
            #     break
            if i < 1000:
                i += 1
                continue
            tree = etree.parse(os.path.join(anno_folder, xml_file))
            img_file_path = os.path.join(self._dataset_path, 'JPEGImages', os.path.splitext(xml_file)[0] + ".jpg")
            root = tree.getroot()
            objects = root.xpath("//object")
            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType.TRAIN)

            for obj in objects:
                box_array = np.array([int(obj.find('bndbox/xmin').text),
                                int(obj.find('bndbox/ymin').text),
                                int(obj.find('bndbox/xmax').text),
                                int(obj.find('bndbox/ymax').text)])
                str = obj.find('name').text
                result_item.append(BoxItem(
                    ori_label=obj.find('name').text,
                    box_array=box_array))

            result.append(result_item)
            # limit=limit-1
            

        return result    
    def ori_label_2_id_map(self) -> dict:
        return {
            'airplane': 255,
            'baseball diamond': 254,
            'basketball court': 253,
            'bridge': 252,
            'crossroad': 251,
            'ground track field': 250,
            'harbor': 249,
            'parking lot': 248,
            'ship': 247,
            'storage tank': 246,
            'T junction': 245,
            'tennis court': 243,
            'vehicle': 242,
            'Mean AP': 241,
        }

   