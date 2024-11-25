# coding: utf-8
import numpy as np
from typing import List
from util.constant import DataType

class BoxItem():
    def __init__(self, ori_label: str, box_array: np.array) -> None:
        self._ori_label = ori_label
        self._box_array = box_array
        self._id = None

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
    def __init__(self, result_list: List[PreprocessResultItem] = None) -> None:
        self._result_list = result_list if result_list is not None else []

    def append(self, item: PreprocessResultItem):
        self._result_list.append(item)

    @property
    def result_list(self):
        return self._result_list
    
    @property
    def ori_labels(self):
        labels = []
        for item in self._result_list:
            labels.extend(item.ori_labels)
        
        return labels
    
    def __len__(self):
        return len(self._result_list)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self._result_list[i]
    
    def __getitem__(self, index):
        return self._result_list[index]