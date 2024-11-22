# coding: utf-8
from __future__ import annotations

import numpy as np

from typing import List

class Mask():
    def __init__(self, img_file_path: str, data: np.ndarray, id: int) -> None:
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


    def update(self, mask: Mask) -> None:
        """合并两个 Mask
        """
        self._data = np.where(self._data > mask.data, self._data, mask.data)
        self._count += 1
        for k, v in mask.get_id_count_map().items():
            if k in self._id_count_map:
                self._id_count_map[k] += v
            else:
                self._id_count_map[k] = v
        

    @property
    def data(self) -> np.ndarray:
        return self._data.astype(np.uint8)
    
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
    def __init__(self, img_file_path: str, mask: Mask):
        self._img_file_path = img_file_path
        self._mask = mask

    @property
    def file_name_without_ext(self) -> str:
        return self._img_file_path.split("/")[-1].split(".")[0]

    @property
    def img_file_path(self) -> str:
        return self._img_file_path

    @property
    def mask(self) -> Mask:
        return self._mask
    
    def get_id_count_map(self) -> dict:
        return self._mask.get_id_count_map()
    
    def get_ids(self) -> List[int]:
        return list(self._mask.get_id_count_map().keys())


class ProcessResult():
    def __init__(self, result_list: List[ProcessResultItem] = None):
        self._result_list = result_list if result_list is not None else []
        self._id_count_map = {}
        for item in self._result_list:
            for k, v in item.get_id_count_map().items():
                if k in self._id_count_map:
                    self._id_count_map[k] += v
                else:
                    self._id_count_map[k] = v
        

    @property
    def result_list(self) -> List[ProcessResultItem]:
        return self._result_list
    
    def append(self, item: ProcessResultItem):
        self._result_list.append(item)
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
        for item in self._result_list:
            yield item

    def __len__(self):
        return len(self._result_list)
    
    def __getitem__(self, index):
        return self._result_list[index]