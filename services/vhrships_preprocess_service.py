# coding: utf-8
import os
import numpy as np

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
from typing import List
from scipy.io import loadmat
from util.box import coco2box

class VhrshipsPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "VHRShips") if dataset_path is None else dataset_path
        anno_label_path = os.path.join(self._dataset_path, "testList_labels.mat")
        anno_data_path = os.path.join(self._dataset_path, "testList_struct.mat")
        anno_labels = loadmat(anno_label_path)
        
        self._labels = [str(item[0]) for item in anno_labels["labels"][0]]
        self._anno_data = loadmat(anno_data_path)['futureStruct']['data'][0][0][0]

    def call(self, limit=-1) -> PreprocessResult:
        ori_img_paths = [str(item[0][0]) for item in self._anno_data[0]]
        img_file_paths = [os.path.join(self._dataset_path, "test_c34", path.split('\\')[-1]) for path in ori_img_paths]
        
        labels_for_img = []
        for labels in [item[0] for item in self._anno_data[2]]:
            tmp = []
            if len(labels) != 0:
                for l in labels:
                    tmp.append(str(l[0][0]))
            labels_for_img.append(tmp)
            
        boxes_for_img = []
        for boxes in [item[0] for item in self._anno_data[1]]:
            tmp = []
            if len(boxes) != 0:
                for b in boxes:
                    tmp.append(b)
            boxes_for_img.append(tmp)
            
        annotations = []  
        for idx, img_file_path in enumerate(img_file_paths):
            anno = {
                'image_path': img_file_path,
            }
            
            box_infos = []
            for y, box in enumerate(boxes_for_img[idx]):
                box_array = box
                ori_label = labels_for_img[idx][y]
                box_info = {
                    'box_array': coco2box(box_array),
                    'ori_label': ori_label
                }
                box_infos.append(box_info)
            anno['box_infos'] = box_infos
            annotations.append(anno)
            
        result = PreprocessResult()
        counter = 0
        for anno in annotations:
            if len(anno['box_infos']) == 0:
                continue
            result_item = PreprocessResultItem(img_file_path=anno['image_path'], data_type=DataType.TRAIN)
            for box_info in anno['box_infos']:
                result_item.append(BoxItem(
                    ori_label=box_info['ori_label'],
                    box_array=box_info['box_array']
                ))
            result.append(result_item)
            counter += 1
            if limit > 0 and counter >= limit:
                break
        return result 


    def ori_label_2_id_map(self) -> dict:
        return {
            'generalCargo_1': 255,
            'ferry_2': 254,
            'roro_3': 253,
            'passanger_4': 252,
            'tug_5': 251,
            'offshore_6': 250,
            'bargePontoon_7': 249,
            'fishing_9': 248,
            'bulkCarrier_10': 247,
            'container_11': 246,
            'oilTanker_12': 245,
            'tanker_13': 244,
            'oreCarrier_14': 243,
            'dredging_15': 242,
            'lpg_17': 241,
            'yatch_18': 240,
            'drill_19': 239,
            'undefined_20': 238,
            'submarine_100': 237,
            'aircraft_200': 236,
            'cruiser_300': 235,
            'destroyer_400': 234,
            'frigate_500': 233,
            'patrolForce_600': 232,
            'landing_700': 231,
            'coastGuard_800': 230,
            'auxilary_900': 229,
            'serviceCraft_1000': 228,
            'other_1100': 227,
            'dredgerReclamation_32': 226,
            'smallPassanger_33': 225,
            'smallBoat_34': 224,
            'coaster_35': 223,
            'floatingDock_36': 222,
        }
    


   