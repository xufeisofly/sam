# coding: utf-8
import os
import sys

import numpy as np
import json

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
from util.logger import logger
from util.box import coco2box
from lxml import etree


class HRSCPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "HRSC2016") if dataset_path is None else dataset_path
        self._image_path = os.path.join(self._dataset_path, "FullDataSet/AllImages")
        self._anno_path = os.path.join(self._dataset_path, "FullDataSet/Annotations")
        self.test_path = os.path.join(self._dataset_path, "ImageSets/test.txt")
        self.train_path = os.path.join(self._dataset_path, "ImageSets/train.txt")
        self.val_path = os.path.join(self._dataset_path, "ImageSets/val.txt")

    def call(self, limit=sys.maxsize) -> PreprocessResult:
        trains = self._get_dataset_from_file(self.train_path)
        tests = self._get_dataset_from_file(self.test_path)
        vals = self._get_dataset_from_file(self.val_path)
        train_result = PreprocessResult()
        val_result = PreprocessResult()
        test_result = PreprocessResult()

        all_xml_files = os.listdir(self._anno_path)
        all_xml_files = [f for f in all_xml_files if f.endswith('.xml')]
        for xml_file in all_xml_files:
            tree = etree.parse(os.path.join(self._anno_path, xml_file))
            img_file_path = os.path.join(self._image_path, os.path.splitext(xml_file)[0] + ".bmp")
            if not os.path.exists(img_file_path):
                print(img_file_path, "is missing! ")
                continue
            root = tree.getroot()
            objects = root.xpath("//HRSC_Object")
            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType.TRAIN)

            for obj in objects:
                box_array = np.array([int(obj.find('box_xmin').text),
                                int(obj.find('box_ymin').text),
                                int(obj.find('box_xmax').text),
                                int(obj.find('box_ymax').text)])

                result_item.append(BoxItem(
                    ori_label=self._get_ori_label_by_token(obj.find('Class_ID').text),
                    box_array=box_array))

            rid = os.path.splitext(xml_file)[0]
            if rid in trains:
                train_result.append(result_item)
            elif rid in vals:
                val_result.append(result_item)
            else:
                test_result.append(result_item)
            limit -= 1
            if limit <= 0: break

        result = PreprocessResult(
            train_result_list=train_result.result_list,
            val_result_list=val_result.result_list,
            test_result_list=test_result.result_list)
        return result

    def _get_dataset_from_file(self, file_path):
        dataset = set()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                dataset.add(line)
        return dataset

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


    def ori_label_2_id_map(self) -> dict:
        return {
            'ship': 255,
            'aircraft carrier': 254,
            'warcraft': 253,
            'merchant ship': 252,
            'Nimitz class aircraft carrier': 251,
            'Enterprise class aircraft carrier': 250,
            'Arleigh Burke class destroyers': 249,
            'WhidbeyIsland class landing craft': 248,
            'Perry class frigate': 247,
            'Sanantonio class amphibious transport dock': 246,
            'Ticonderoga class cruiser': 245,
            'Kitty Hawk class aircraft carrier': 244,
            'Admiral Kuznetsov aircraft carrier': 243,
            'Abukuma-class destroyer escort': 242,
            'Austen class amphibious transport dock': 241,
            'Tarawa-class amphibious assault ship': 240,
            'USS Blue Ridge (LCC-19)': 239,
            'Container ship': 238,
            'OXo': 237,
            'Car carrier': 236,
            'Hovercraft': 235,
            'yacht': 234,
            'Container ship2': 233,
            'Cruise ship': 232,
            'submarine': 231,
            'lute': 230,
            'Medical ship': 229,
            'Car carrier2': 228,
            'Ford-class aircraft carriers': 227,
            'Midway-class aircraft carrier': 226,
            'Invincible-class aircraft carrier': 225,
        }

    def _get_ori_label_by_token(self, label_token: int) -> str:
        token2ori = {
            '100000001': 'ship',
            '100000002': 'aircraft carrier',
            '100000003': 'warcraft',
            '100000004': 'merchant ship',
            '100000005': 'Nimitz class aircraft carrier',
            '100000006': 'Enterprise class aircraft carrier',
            '100000007': 'Arleigh Burke class destroyers',
            '100000008': 'WhidbeyIsland class landing craft',
            '100000009': 'Perry class frigate',
            '100000010': 'Sanantonio class amphibious transport dock',
            '100000011': 'Ticonderoga class cruiser',
            '100000012': 'Kitty Hawk class aircraft carrier',
            '100000013': 'Admiral Kuznetsov aircraft carrier',
            '100000014': 'Abukuma-class destroyer escort',
            '100000015': 'Austen class amphibious transport dock',
            '100000016': 'Tarawa-class amphibious assault ship',
            '100000017': 'USS Blue Ridge (LCC-19)',
            '100000018': 'Container ship',
            '100000019': 'oXo',
            '100000020': 'Car carrier',
            '100000022': 'Hovercraft',
            '100000024': 'yacht',
            '100000025': 'Container ship2',
            '100000026': 'Cruise ship',
            '100000027': 'submarine',
            '100000028': 'lute',
            '100000029': 'Medical ship',
            '100000030': 'Car carrier2',
            '100000031': 'Ford-class aircraft carriers',
            '100000032': 'Midway-class aircraft carrier',
            '100000033': 'Invincible-class aircraft carrier'
        }
        return token2ori[label_token]


