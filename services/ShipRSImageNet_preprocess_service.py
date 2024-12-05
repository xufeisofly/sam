# coding: utf-8
import os
import numpy as np
from util.logger import logger
from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
from typing import List
from lxml import etree

class ShipRSImageNetPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join("/root/liuh/sam/origin_data/", "ShipRSImageNet/ShipRSImageNet_V1/VOC_Format") if dataset_path is None else dataset_path
        self.train_set = self.get_set_from_txt('/root/liuh/sam/origin_data/ShipRSImageNet/ShipRSImageNet_V1/VOC_Format/ImageSets/train.txt')
        self.val_set = self.get_set_from_txt('/root/liuh/sam/origin_data/ShipRSImageNet/ShipRSImageNet_V1/VOC_Format/ImageSets/val.txt')
        self.test_set = self.get_set_from_txt('/root/liuh/sam/origin_data/ShipRSImageNet/ShipRSImageNet_V1/VOC_Format/ImageSets/test.txt')
    def get_set_from_txt(self, txt:str) -> list[str]:
        with open(txt, 'r', encoding='utf-8') as file:
        # 按行读取并去除空格
            lines = [line.strip().split('.')[0] for line in file.readlines()]
            return lines
    def ret_datatype(self, num:str) -> DataType:
        if num in self.train_set:
            return DataType.TRAIN
        elif num in self.val_set:
            return DataType.VAL
        elif num in self.test_set:
            return DataType.TEST
        else:
            logger.warning("unknown data type. set to train")
            return DataType.TRAIN
    def call(self, limit=-1) -> PreprocessResult:
        anno_folder = os.path.join(self._dataset_path, "Annotations/")
        all_xml_files = os.listdir(anno_folder)
        all_xml_files = [f for f in all_xml_files if f.endswith('.xml') and not f.startswith('._')]
        all_xml_files.sort() # len = 2748

        result = PreprocessResult()
        # i = 0
        for xml_file in all_xml_files:
            if limit == 0:
                break
            # if i < 1000:
            #     i += 1
            #     continue
            tree = etree.parse(os.path.join(anno_folder, xml_file))
            img_file_path = os.path.join(self._dataset_path, 'JPEGImages', os.path.splitext(xml_file)[0] + ".bmp")
            if os.path.exists(img_file_path) == False:
                print(f"{img_file_path} 不存在")
                
            root = tree.getroot()
            objects = root.xpath("//object")
            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=self.ret_datatype(os.path.splitext(xml_file)[0]))

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
            limit=limit-1
        return result    
    def ori_label_2_id_map(self) -> dict:
        return {
            'Other Ship': 255,
            'Other Warship': 254,
            'Submarine': 253,
            'Other Aircraft Carrier': 252,
            'Enterprise': 251,
            'Nimitz': 250,
            'Midway': 249,
            'Ticonderoga': 248,
            'Other Destroyer': 247,
            'Atago DD': 246,
            'Arleigh Burke DD': 245,
            'Hatsuyuki DD': 244,
            'Hyuga DD': 243,
            'Asagiri DD': 242,
            'Other Frigate': 241,
            'Perry FF': 240,
            'Patrol': 239,
            'Other Landing': 238,
            'YuTing LL': 237,
            'YuDeng LL': 236,
            'YuDao LL': 235,
            'YuZhao LL': 234,
            'Austin LL': 233,
            'Osumi LL': 232,
            'Wasp LL': 231,
            'LSD 41 LL': 230,
            'LHA LL': 229,
            'Commander': 228,
            'Other Auxiliary Ship': 227,
            'Medical Ship': 226,
            'Test Ship':225,
            'Training Ship':224,
            'AOE':223,
            'Masyuu AS':222,
            'Sanantonio AS':221,
            'EPF':220,
            'Other Merchant':219,
            'Container Ship':218,
            'RoRo':217,
            'Cargo':216,
            'Barge':215,
            'Tugboat':214,
            'Ferry':213,
            'Yacht':212,
            'Sailboat':211,
            'Fishing Vessel':210,
            'Oil Tanker':209,
            'Hovercraft':208,
            'Motorboat':207,
            'Dock':206,           
        }