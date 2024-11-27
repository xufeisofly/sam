# coding: utf-8
import __future__
import os
import json

from schemas.process_result import ProcessResult, ProcessResultItem
from util.constant import OUTPUT_DIR, DataType
from typing import Tuple
from PIL import Image



class OutputService():
    def __init__(self, dataset_name: str='default'):
        self._dataset_name = dataset_name
        self._output_dir = os.path.join(OUTPUT_DIR, self._dataset_name)
        self._output_ann_dir = os.path.join(self._output_dir, "ann_dir")
        self._output_img_dir = os.path.join(self._output_dir, "img_dir")

        self._train_ann_dir = os.path.join(self._output_ann_dir, "train")
        self._val_ann_dir = os.path.join(self._output_ann_dir, "val")
        self._test_ann_dir = os.path.join(self._output_ann_dir, "test")
        self._train_img_dir = os.path.join(self._output_img_dir, "train")
        self._val_img_dir = os.path.join(self._output_img_dir, "val")
        self._test_img_dir = os.path.join(self._output_img_dir, "test")
        self._origin_data_dir = os.path.join(self._output_dir, "origin_data")
        
        self._detection_dir = os.path.join(self._output_dir, "detection_data")
        self._detection_ann_dir = os.path.join(self._detection_dir, "annotations")
        
        for folder in [self._output_ann_dir, self._output_img_dir, self._train_ann_dir,
                       self._val_ann_dir, self._test_ann_dir,
                       self._train_img_dir, self._val_img_dir, self._test_img_dir,
                       self._detection_ann_dir, 
                       self._origin_data_dir]:
            os.makedirs(folder, exist_ok=True)

    def clear_all(self):
        for folder in [self._train_ann_dir, self._val_ann_dir, self._test_ann_dir,
                       self._train_img_dir, self._val_img_dir,
                       self._test_img_dir, self._detection_ann_dir, 
                       ]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))


    def call(self, data: ProcessResult, ori_label_2_id_map: dict):
        self.clear_all()
        reordered_result = self.reorder_result_data_type(data)
        # 保存语义分割数据
        self.save_masks(reordered_result)
        # 保存标签映射
        self.save_type_map(ori_label_2_id_map)
        # TODO 保存目标检测数据
        self.save_detection_data(reordered_result, ori_label_2_id_map)
        

    def reorder_result_data_type(self, result: ProcessResult) -> ProcessResult:
        """将 result 按比例分为训练集和验证集
        """
        new_result = ProcessResult()
        if len(result.val_result_list) == 0:
            train_result, val_result = self._extract_val_from_train(result)
        else:
            train_result = ProcessResult(train_result_list=result.train_result_list)
            val_result = ProcessResult(val_result_list=result.val_result_list)
            
        return ProcessResult(train_result_list=train_result.result_list, 
                             val_result_list=val_result.result_list,
                             test_result_list=result.test_result_list)
        
    
    def _extract_val_from_train(self, result: ProcessResult) -> Tuple[ProcessResult, ProcessResult]:
        train_result = ProcessResult()
        val_result = ProcessResult()

        id_counter = {}
        expected_id_count = self.get_val_expected_id_count_map(result)
        val_is_enough = False
        for result_item in result.train_result_list:
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
    
    def save_masks(self, result: ProcessResult):
        self.save_to_ann_dir(result)
        self.save_to_img_dir(result)


    def save_to_ann_dir(self, result: ProcessResult):
        for result_item in result.train_result_list:
            self._save_mask_result_item_tif(result_item, self._train_ann_dir)
        for result_item in result.val_result_list:
            self._save_mask_result_item_tif(result_item, self._val_ann_dir)
        for result_item in result.test_result_list:
            self._save_mask_result_item_tif(result_item, self._test_ann_dir)
            
    def save_to_img_dir(self, result: ProcessResult):
        for result_item in result.train_result_list:
            self._save_ori_image_tif(result_item, self._train_img_dir)
        for result_item in result.val_result_list:
            self._save_ori_image_tif(result_item, self._val_img_dir)
        for result_item in result.test_result_list:
            self._save_ori_image_tif(result_item, self._test_img_dir)

    def save_type_map(self, ori_label_2_id_map: dict):
        with open(os.path.join(self._output_dir, 'type_map.json'), 'w') as f:
            json.dump(ori_label_2_id_map, f, indent=4)
        
    def _save_mask_result_item_tif(self, result_item: ProcessResultItem, dir):
        image = Image.fromarray(result_item.mask.data)
        file = os.path.join(dir, result_item.file_name_without_ext + ".tif")
        image.save(file)
        
    def _save_ori_image_tif(self, result_item: ProcessResultItem, dir):
        image = Image.open(result_item.img_file_path)
        file = os.path.join(dir, result_item.file_name_without_ext + ".tif")
        image.save(file)       

    def save_detection_data(self, data: ProcessResult, ori_label_2_id_map: dict):
        """保存 detection 数据

        Args:
            data (ProcessResult): _description_
        """
        train_annotations = []
        val_annotations = []
        test_annotations = []
 
        for result_item in data:
            for box_item in result_item.mask.box_items:
                ann = {
                    'image': self._get_detection_image_path(result_item),
                    'category_id': box_item.id,
                    'category_name': box_item.ori_label,
                    'bbox': box_item.box_array_coco,
                }

                if result_item.data_type == DataType.TRAIN:
                    train_annotations.append(ann)
                elif result_item.data_type == DataType.VAL:
                    val_annotations.append(ann)
                else:
                    test_annotations.append(ann)
        
        categories = []          
        for id, name in ori_label_2_id_map.items():
            categories.append({
                'id': id,
                'name': name,
            })
        
        if len(train_annotations) > 0:
            with open(os.path.join(self._detection_ann_dir, 'train.json'), 'w') as f:
                json.dump({'annotations': train_annotations, 'categories': categories}, f, indent=4)
        if len(val_annotations) > 0:
            with open(os.path.join(self._detection_ann_dir, 'val.json'), 'w') as f:
                json.dump({'annotations': val_annotations, 'categories': categories}, f, indent=4)            
        if len(test_annotations) > 0:
            with open(os.path.join(self._detection_ann_dir, 'test.json'), 'w') as f:
                json.dump({'annotations': test_annotations, 'categories': categories}, f, indent=4)
                        
    def _get_detection_image_path(self, result_item: ProcessResultItem):
        return result_item.data_type.value + '/' + result_item.file_name_without_ext + '.tif'