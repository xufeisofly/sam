# coding: utf-8
from __future__ import annotations

import numpy as np

from typing import List
from schemas.preprocess_result import BoxItem
from util.constant import DataType
from util.box import calculate_island_area
from util.logger import logger


class Mask():
    def __init__(self, img_file_path: str, data: np.ndarray, id: int, box_items: List[BoxItem]=None) -> None:
        """
        data: np.array([], dtype=np.boolean)
        """
        if data.ndim != 2:
            raise ValueError("data must be a 2D numpy array")
        if id <= 0:
            raise ValueError("id must be greater than zero")
        self._data = data.astype(int) * id
        self._count = 1
        self._img_file_path = img_file_path
        self._id_count_map = {
            id: 1,
        }
        self._box_items = box_items if box_items is not None else []


    def update(self, mask: Mask) -> None:
        """合并两个 Mask
        """
        mask_new_region = (mask.data != 0) & (self._data == 0)
        overlap_region = (mask.data > 0) & (self._data > 0)

        # 更新新的区域
        self._data[mask_new_region] = mask.data[mask_new_region]
        
        if overlap_region.any():
            # 提取第一个重叠的位置
            overlap_positions = np.array(list(zip(*np.where(overlap_region))))
            first_row, first_col = overlap_positions[0]
            old_id = self._data[first_row, first_col]
            new_id = mask.data[first_row, first_col]

            # 缓存小岛面积
            old_id_size = calculate_island_area(self._data, first_row, first_col)
            new_id_size = np.sum(mask.data == new_id)

            # 如果旧小岛面积大于新小岛，更新重叠区域
            if old_id_size > new_id_size:
                self._data[overlap_region] = new_id
            
            logger.info(f"===== -> {new_id}")
        
        # self._data = np.where(self._data > mask.data, self._data, mask.data)
        self._count += 1
        for k, v in mask.get_id_count_map().items():
            if k in self._id_count_map:
                self._id_count_map[k] += v
            else:
                self._id_count_map[k] = v
        self._box_items.extend(mask.box_items)
        

    @property
    def data(self) -> np.ndarray:
        return self._data.astype(np.uint8)
    
    @property
    def box_items(self) -> List[BoxItem]:
        return self._box_items
    
    def get_id_count_map(self) -> dict:
        """
        获取每个 id 的数量
        """
        return self._id_count_map
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def img_file_path(self) -> str:
        return self._img_file_path
    
class ProcessResultItem():
    def __init__(self, img_file_path: str, mask: Mask, data_type=DataType.TRAIN, mask_img_file_path:str = None):
        self._img_file_path = img_file_path
        self._mask_img_file_path = mask_img_file_path if mask_img_file_path is not None else img_file_path
        self._mask = mask
        self._data_type = data_type # 原始数据类型 train, validation, test

    @property
    def file_name_without_ext(self) -> str:
        return self._img_file_path.split("/")[-1].split(".")[0]
    
    @property
    def mask_file_name_without_ext(self) -> str:
        return self._mask_img_file_path.split("/")[-1].split(".")[0]

    @property
    def img_file_path(self) -> str:
        return self._img_file_path
    
    @property
    def mask_img_file_path(self) -> str:
        return self._mask_img_file_path

    @property
    def mask(self) -> Mask:
        return self._mask
    
    @property
    def data_type(self) -> DataType:
        return self._data_type
    
    def get_id_count_map(self) -> dict:
        return self._mask.get_id_count_map()
    
    def get_ids(self) -> List[int]:
        return list(self._mask.get_id_count_map().keys())


class ProcessResult():
    def __init__(self, train_result_list: List[ProcessResultItem] = None,
                 val_result_list: List[ProcessResultItem] = None,
                 test_result_list: List[ProcessResultItem] = None):
        self._train_result_list = train_result_list if train_result_list is not None else []
        self._val_result_list = val_result_list if val_result_list is not None else []
        self._test_result_list = test_result_list if test_result_list is not None else []
        self._id_count_map = {}
        for item in self.result_list:
            for k, v in item.get_id_count_map().items():
                if k in self._id_count_map:
                    self._id_count_map[k] += v
                else:
                    self._id_count_map[k] = v
  
    @property
    def train_result_list(self) -> List[ProcessResultItem]:
        return self._train_result_list

    @property
    def val_result_list(self) -> List[ProcessResultItem]:
        return self._val_result_list

    @property
    def test_result_list(self) -> List[ProcessResultItem]:
        return self._test_result_list

    @property
    def result_list(self) -> List[ProcessResultItem]:
        return self._train_result_list + self._val_result_list + self._test_result_list
    
    def append(self, item: ProcessResultItem):
        if item.data_type == DataType.TRAIN:
            self._train_result_list.append(item)
        elif item.data_type == DataType.VAL:
            self._val_result_list.append(item)
        elif item.data_type == DataType.TEST:
            self._test_result_list.append(item)
        else:
            raise ValueError("Invalid data type")

        for k, v in item.get_id_count_map().items():
            if k in self._id_count_map:
                self._id_count_map[k] += v
            else:
                self._id_count_map[k] = v

    def get_id_count_map(self) -> dict:
        return self._id_count_map
    
    def all_ids(self):
        return list(self._id_count_map.keys())

    def __iter__(self):
        for item in self.result_list:
            yield item

    def __len__(self):
        return len(self.result_list)
    
    def __getitem__(self, index):
        return self.result_list[index]