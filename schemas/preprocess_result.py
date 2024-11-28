# coding: utf-8
import numpy as np
from typing import List
from util.constant import DataType

class BoxItem():
    def __init__(self, ori_label: str, box_array: np.array) -> None:
        self._ori_label = ori_label
        self._box_array = box_array
        self._id = None
        self._confidence_value = None

    @property
    def ori_label(self):
        return self._ori_label

    @property
    def box_array(self):
        return self._box_array
    
    @property
    def box_array_coco(self):
        return [int(self._box_array[0]), int(self._box_array[1]), int(self._box_array[2]-self._box_array[0]), int(self._box_array[3]-self._box_array[1])]
    
    @property
    def id(self):
        return self._id
    
    def set_id(self, value):
        self._id = value
        
    @property
    def confidence_value(self):
        return self._confidence_value
    
    def set_confidence_value(self, value):
        self._confidence_value = value
        
    def box_string(self):
        return '-'.join([str(x) for x in self._box_array])        


class PreprocessResultItem():
    def __init__(self, img_file_path: str, box_items: List[BoxItem] = None, data_type: DataType=DataType.TRAIN) -> None:
        self._img_file_path = img_file_path
        self._box_items = box_items if box_items is not None else []
        self._data_type = data_type

    def append(self, item: BoxItem):
        self._box_items.append(item)

    @property
    def img_file_path(self):
        return self._img_file_path
    
    @property
    def box_items(self):
        return self._box_items
    
    @property
    def ori_labels(self):
        return [item.ori_label for item in self._box_items]
    
    @property
    def data_type(self) -> DataType:
        return self._data_type


class PreprocessResult():
    def __init__(self, train_result_list: List[PreprocessResultItem] = None,
                 val_result_list: List[PreprocessResultItem] = None,
                 test_result_list: List[PreprocessResultItem] = None) -> None:
        self._train_result_list = train_result_list if train_result_list is not None else []
        self._val_result_list = val_result_list if val_result_list is not None else []
        self._test_result_list = test_result_list if test_result_list is not None else []

    def append(self, item: PreprocessResultItem):
        if item.data_type == DataType.TRAIN:
            self._train_result_list.append(item)
        elif item.data_type == DataType.VAL:
            self._val_result_list.append(item)
        elif item.data_type == DataType.TEST:
            self._test_result_list.append(item)
            
    @property
    def train_result_list(self):
        return self._train_result_list
    
    @property
    def val_result_list(self):
        return self._val_result_list
    
    @property
    def test_result_list(self):
        return self._test_result_list
    
    @property
    def result_list(self):
        return self._train_result_list + self._val_result_list + self._test_result_list
    
    @property
    def ori_labels(self):
        labels = []
        for item in self.result_list:
            labels.extend(item.ori_labels)
        
        return labels
    
    def __len__(self):
        return len(self.result_list)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.result_list[i]
    
    def __getitem__(self, index):
        return self.result_list[index]