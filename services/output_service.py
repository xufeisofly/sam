# coding: utf-8
import os

from schemas.process_result import ProcessResult, ProcessResultItem
from util.constant import OUTPUT_DIR
from typing import Tuple
from PIL import Image


class OutputService():
    def __init__(self, dataset_name: str='default'):
        self._dataset_name = dataset_name
        self._output_dir = os.path.join(OUTPUT_DIR, self._dataset_name)
        self._output_ann_dir = os.path.join(self._output_dir, "ann_dir")
        self._output_img_dir = os.path.join(self._output_dir, "img_dir")
        os.makedirs(self._output_ann_dir, exist_ok=True)
        os.makedirs(self._output_img_dir, exist_ok=True)

        self._train_ann_dir = os.path.join(self._output_ann_dir, "train")
        self._val_ann_dir = os.path.join(self._output_ann_dir, "val")
        self._train_img_dir = os.path.join(self._output_img_dir, "train")
        self._val_img_dir = os.path.join(self._output_img_dir, "val")

        os.makedirs(self._train_ann_dir, exist_ok=True)
        os.makedirs(self._val_ann_dir, exist_ok=True)
        os.makedirs(self._train_img_dir, exist_ok=True)
        os.makedirs(self._val_img_dir, exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "origin_data"), exist_ok=True)

    def clear_all_images(self):
        for folder in [self._train_ann_dir, self._val_ann_dir, self._train_img_dir, self._val_img_dir]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

    def seperate_train_and_val_result(self, result: ProcessResult) -> Tuple[ProcessResult, ProcessResult]:
        """将 result 按比例分为训练集和验证集
        """
        train_result = ProcessResult()
        val_result = ProcessResult()

        id_counter = {}
        expected_id_count = self.get_val_expected_id_count_map(result)
        val_is_enough = False
        for result_item in result:
            if not val_is_enough:
                val_result.append(result_item)
                for id, cnt in result_item.get_id_count_map().items():
                    if not id_counter.get(id, 0):
                        id_counter[id] = 0
                    id_counter[id] += cnt
                if self._check_all_ids_cnt_enough(id_counter, expected_id_count):
                    val_is_enough = True
            else:
                train_result.append(result_item)

        return train_result, val_result
    
    def _check_all_ids_cnt_enough(self, id_counter: dict, expected_id_count: dict):
        for id, cnt in expected_id_count.items():
            if id_counter.get(id, 0) < cnt:
                return False
        return True

    
    def get_val_expected_id_count_map(self, result: ProcessResult):
        expected_id_count = {}
        for id, cnt in result.get_id_count_map().items():
            if not expected_id_count.get(id, 0):
                expected_id_count[id] = 0
            expected_id_count[id] = int(min(cnt * 0.3, 30))
        return expected_id_count


    def save_to_ann_dir(self, train_result: ProcessResult, val_result: ProcessResult):
        for result_item in train_result:
            self._save_result_item_tif(result_item, self._train_ann_dir)
        for result_item in val_result:
            self._save_result_item_tif(result_item, self._val_ann_dir)
        
    def _save_result_item_tif(self, result_item: ProcessResultItem, dir):
        image = Image.fromarray(result_item.mask.data)
        file = os.path.join(dir, result_item.file_name_without_ext + ".tif")
        image.save(file)

    def save_type_map(self, data: ProcessResult):
        pass

    def save_img_dir(self, data: ProcessResult):
        pass