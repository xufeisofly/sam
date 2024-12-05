# coding: utf-8
import __future__
import os
import json
import random
import shutil

from schemas.process_result import Mask, ProcessResult, ProcessResultItem
from util.constant import OUTPUT_DIR, DataType
from typing import Tuple, List
from PIL import Image
from schemas.preprocess_result import BoxItem



class OutputService():
    def __init__(self, dataset_name: str='default'):
        self._dataset_name = dataset_name
        self._output_dir = os.path.join(OUTPUT_DIR, self._dataset_name)
        os.makedirs(self._output_dir, exist_ok=True)

    def build_output(self):
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
        
        for folder in [self._output_dir, self._output_ann_dir, self._output_img_dir, self._train_ann_dir,
                       self._val_ann_dir, self._test_ann_dir,
                       self._train_img_dir, self._val_img_dir, self._test_img_dir,
                       self._detection_ann_dir, 
                       self._origin_data_dir]:
            os.makedirs(folder, exist_ok=True)
        
    def clear_output(self):
        shutil.rmtree(self._output_dir)
        self.build_output()
        # for folder in [self._train_ann_dir, self._val_ann_dir, self._test_ann_dir,
        #                self._train_img_dir, self._val_img_dir,
        #                self._test_img_dir, self._detection_ann_dir, 
        #                ]:
        #     for f in os.listdir(folder):
        #         os.remove(os.path.join(folder, f))


    def save_rest(self, data: ProcessResult, ori_label_2_id_map: dict):
        # 保存语义分割数据
        # self.save_masks(reordered_result)
        self._save_to_img_dir(data)
        # 保存标签映射
        self._save_type_map(ori_label_2_id_map)
        # 保存目标检测数据
        self.save_detection_data(data, ori_label_2_id_map)      
        

    def classify_result(self, result: ProcessResult):
        """将 result 按比例分为训练集和验证集
        """
        if len(result.val_result_list) == 0:
            train_result, val_result = self._extract_val_from_train(result)
        else:
            train_result = ProcessResult(train_result_list=result.train_result_list)
            val_result = ProcessResult(val_result_list=result.val_result_list)
            
        data = ProcessResult(train_result_list=train_result.result_list, 
                             val_result_list=val_result.result_list,
                             test_result_list=result.test_result_list)
        
        classify_dict = {}
        for item in data.train_result_list:
            classify_dict[item.file_name_without_ext] = DataType.TRAIN.value
        for item in data.val_result_list:
            classify_dict[item.file_name_without_ext] = DataType.VAL.value
        for item in data.test_result_list:
            classify_dict[item.file_name_without_ext] = DataType.TEST.value

        return data, classify_dict
    
    def classify_result_by_dict(self, result: ProcessResult, classify_dict: dict) -> ProcessResult:
        train_result_items = []
        val_result_items = []
        test_result_items = []
        for result_item in result.result_list:
            type = classify_dict.get(result_item.file_name_without_ext, DataType.TRAIN.value)
            if type == DataType.TRAIN.value:
                train_result_items.append(result_item)
            elif type == DataType.VAL.value:
                val_result_items.append(result_item)
            elif type == DataType.TEST.value:
                test_result_items.append(result_item)
                
        return ProcessResult(train_result_list=train_result_items, 
                             val_result_list=val_result_items,
                             test_result_list=test_result_items)
        
    
    def _extract_val_from_train(self, result: ProcessResult) -> Tuple[ProcessResult, ProcessResult]:
        train_result = ProcessResult()
        val_result = ProcessResult()

        id_counter = {}
        expected_id_count = self.get_val_expected_id_count_map(result)
        val_is_enough = False
        # 打乱顺序，以便于后续的验证集采样
        random.shuffle(result.train_result_list)
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
    
    def save_masks(self, result: ProcessResult, ori_label_2_id_map: dict):
        self._save_to_ann_dir(result)
        self._save_sam_result_data(result, ori_label_2_id_map)  


    def _save_to_ann_dir(self, result: ProcessResult):
        for result_item in result.train_result_list:
            self._save_mask_result_item_tif(result_item, self._train_ann_dir)
        for result_item in result.val_result_list:
            self._save_mask_result_item_tif(result_item, self._val_ann_dir)
        for result_item in result.test_result_list:
            self._save_mask_result_item_tif(result_item, self._test_ann_dir)
            
    def _save_to_img_dir(self, result: ProcessResult):
        for result_item in result.train_result_list:
            self._save_ori_image_tif(result_item, self._train_img_dir)
        for result_item in result.val_result_list:
            self._save_ori_image_tif(result_item, self._val_img_dir)
        for result_item in result.test_result_list:
            self._save_ori_image_tif(result_item, self._test_img_dir)

    def _save_type_map(self, ori_label_2_id_map: dict):
        with open(os.path.join(self._output_dir, 'type_map.json'), 'w') as f:
            json.dump(ori_label_2_id_map, f, indent=4)
        
    def _save_mask_result_item_tif(self, result_item: ProcessResultItem, dir):
        if result_item.mask is Mask.EMPTY:
            return
        image = Image.fromarray(result_item.mask.data)
        file = os.path.join(dir, result_item.mask.mask_file_name_without_ext + ".tif")
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
        self._save_detection_data(data.train_result_list, ori_label_2_id_map, DataType.TRAIN)
        self._save_detection_data(data.val_result_list, ori_label_2_id_map, DataType.VAL)
        self._save_detection_data(data.test_result_list, ori_label_2_id_map, DataType.TEST)
                
    def _save_detection_data(self, result_items: List[ProcessResultItem], ori_label_2_id_map: dict, data_type: DataType):
        annotations = []
        
        for result_item in result_items:
            for box_item in result_item.mask.box_items:
                ann = {
                    'image': self._get_detection_image_path(result_item, data_type=data_type),
                    'category_id': box_item.id,
                    'category_name': box_item.ori_label,
                    'bbox': box_item.box_array_coco,
                }
                
                annotations.append(ann)
        
        categories = []          
        for id, name in ori_label_2_id_map.items():
            categories.append({
                'id': id,
                'name': name,
            })
        
        if len(annotations) > 0:
            with open(os.path.join(self._detection_ann_dir, data_type.value + '.json'), 'w') as f:
                json.dump({'annotations': annotations, 'categories': categories}, f, indent=4)
        
    def _save_sam_result_data(self, data: ProcessResult, ori_label_2_id_map: dict):
        self._save_sam_result_data_item(data.train_result_list, DataType.TRAIN)
        self._save_sam_result_data_item(data.val_result_list, DataType.VAL)
        self._save_sam_result_data_item(data.test_result_list, DataType.TEST)
        
    def _save_sam_result_data_item(self, result_items: List[ProcessResultItem], data_type: DataType):
        annotations = []
        
        for result_item in result_items:
            image = Image.open(result_item.img_file_path)
            for box_item in result_item.mask.box_items:
                ann = {
                    'image': self._get_detection_image_path(result_item, data_type=data_type),
                    'image_size': [image.width, image.height],
                    'confidence_value': str(box_item.confidence_value),
                    'bbox': box_item.box_array_coco,
                }
                
                annotations.append(ann)
                
        if len(annotations) > 0:
            json_file = os.path.join(self._output_ann_dir, data_type.value + '.json')
            json_data = {}
            if os.path.exists(json_file):                
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    json_data['annotations'].extend(annotations)
            else:
                json_data = {'annotations': annotations}
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=4)
                        
    def _get_detection_image_path(self, result_item: ProcessResultItem, data_type: DataType):
        return data_type.value + '/' + result_item.file_name_without_ext + '.tif'