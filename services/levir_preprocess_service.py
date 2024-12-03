# coding: utf-8
import os
import sys

import numpy as np
from torch.backends.cudnn.rnn import init_dropout_state
from torchvision.ops import box_area

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
import cv2


class LevirPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "LEVIR") if dataset_path is None else dataset_path
        self._data_path = os.path.join(self._dataset_path, "imageWithLabel")

    def call(self, limit=-1) -> PreprocessResult:
        if limit < 0: limit = sys.maxsize

        all_data_files = os.listdir(self._data_path)
        all_box_files = [f for f in all_data_files if f.endswith('.txt')]

        result = PreprocessResult()
        for box_file in all_box_files:
            box_path = os.path.join(self._data_path, box_file)
            img_file_path = os.path.join(self._data_path, os.path.splitext(box_file)[0] + ".jpg")
            if not os.path.exists(img_file_path):
                print(img_file_path, "is missing! ")
                continue
            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType.TRAIN)
            with open(box_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    data = line.split(' ')
                    if len(data) != 5: continue
                    data = list(map(int, data))
                    box_array = np.array([data[1], data[2], data[3], data[4]])
                    result_item.append(BoxItem(
                        ori_label=self._get_ori_label_by_token(data[0]),
                        box_array=box_array))

            if not result_item.box_items:
                continue
            result.append(result_item)
            limit -= 1
            if limit <= 0: break
        return result

    def ori_label_2_id_map(self) -> dict:
        return {
            'plane': 255,
            'ship': 254,
            'oilpot':253
        }

    def _get_ori_label_by_token(self, label_token: int) -> str:
        token2ori = {
            1: 'plane',
            2: 'ship',
            3: 'oilpot'
        }
        return token2ori[label_token]

    def test(self):
        all_image_files = os.listdir(self._data_path)
        all_box_files = [f for f in all_image_files if f.endswith('.txt')]
        ids = set()
        for box_file in all_box_files:
            box_path = os.path.join(self._data_path, box_file)
            with open(box_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    data = line.split(' ')
                    if len(data) != 5: continue
                    data = list(map(int, data))
                    ids.add(data[0])
        print(ids)

if __name__ == "__main__":
    service = LevirPreprocessService()
    service.test()
