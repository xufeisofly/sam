# coding: utf-8
from __future__ import annotations

import numpy as np
import os
import pickle
import weakref
from pathlib import Path

from typing import List
from schemas.preprocess_result import BoxItem
from util.constant import DataType, ROOT_DIR
from util.box import box2coco, calculate_island_area
from util.logger import logger
from util.file import get_file_name_without_ext


class Mask():
    EMPTY = None
    
    def __init__(self, img_file_path: str='', data: np.ndarray=None, id: int=-1, box_items: List[BoxItem]=None, mask_img_file_path: str=None) -> None:
        """
        data: np.array([], dtype=np.boolean)
        """
        if data is not None and data.ndim != 2:
            raise ValueError("data must be a 2D numpy array")
        if data is not None:
            self._data = data.astype(int) * id
        else:
            self._data = None
        self._count = 1
        self._img_file_path = img_file_path
        self._mask_img_file_path = mask_img_file_path if mask_img_file_path is not None else img_file_path
        self._id_count_map = {}
        if id >= 0:
            self._id_count_map[id] = 1
        self._box_items = box_items if box_items is not None else []


    def update(self, mask: Mask) -> None:
        """合并两个 Mask
        """
        self._count += 1
        for k, v in mask.get_id_count_map().items():
            if k in self._id_count_map:
                self._id_count_map[k] += v
            else:
                self._id_count_map[k] = v
        self._box_items.extend(mask.box_items)
        
        if mask.data is None and self._data is None:
            return
        
        mask_new_region = (mask.data != 0) & (self._data == 0)
        overlap_region = (mask.data > 0) & (self._data > 0) & (self._data != mask.data)

        # 更新新的区域
        self._data[mask_new_region] = mask.data[mask_new_region]
        
        if overlap_region.any():
            logger.debug(f"Overlapped")
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
        # else:
            # logger.debug(f"No overlapped")
        
        # self._data = np.where(self._data > mask.data, self._data, mask.data)

        
        
    @property
    def mask_img_file_path(self) -> str:
        return self._mask_img_file_path
        

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            return None
        return self._data.astype(np.uint8)
    
    def has_data(self) -> bool:
        return self.data is not None
    
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
    
    @property
    def mask_img_file_path(self) -> str:
        return self._mask_img_file_path
    
    @property
    def mask_file_name_without_ext(self) -> str:
        return get_file_name_without_ext(self._mask_img_file_path)


Mask.EMPTY = Mask()
    
class ProcessResultItem():
    def __init__(self, img_file_path: str, mask: Mask=Mask.EMPTY, data_type=DataType.TRAIN, disk_for_mask=False):
        if mask is None:
            raise ValueError('mask cannot be None')
        self._img_file_path = img_file_path
        if not disk_for_mask:
            self._mask = mask
        else:
            self._mask_file_path = self._generate_mask_file_path(img_file_path)
            self._save_mask_to_disk(mask)
            self._register_cleanup()
        self._use_disk = disk_for_mask
        self._data_type = data_type # 原始数据类型 train, validation, test
        
    @staticmethod
    def _generate_mask_file_path(img_file_path: str) -> str:
        """
        根据 img_file_path 生成 mask 文件路径。
        例如，将原始文件扩展名替换为 '.mask.pkl'。
        """
        return str(Path(
            os.path.join(ROOT_DIR, 'tmp', get_file_name_without_ext(img_file_path)))
            .with_suffix('.mask.pkl'))
    
    def _save_mask_to_disk(self, mask: 'Mask'):
        """
        将 Mask 对象序列化并保存到硬盘。
        """
        with open(self._mask_file_path, 'wb') as f:
            pickle.dump(mask, f)
        logger.debug(f"Mask 已保存到磁盘: {self._mask_file_path}")
        
    def _cleanup(self):
        """
        删除硬盘上的 mask 文件。
        """
        if Path(self._mask_file_path).exists():
            try:
                os.remove(self._mask_file_path)
                logger.debug(f"删除临时文件: {self._mask_file_path}")
            except OSError as e:
                logger.debug(f"删除文件失败: {e}")
        
    def _register_cleanup(self):
        """
        注册对象回收时的清理函数。
        使用弱引用管理清理，避免阻止对象回收。
        """
        weakref.finalize(self, self._cleanup)

    @property
    def file_name_without_ext(self) -> str:
        return get_file_name_without_ext(self._img_file_path)
    

    @property
    def img_file_path(self) -> str:
        return self._img_file_path
    

    @property
    def mask(self) -> Mask:
        if self._use_disk:
            return self._load_mask_from_disk()
        return self._mask
    
    def _load_mask_from_disk(self) -> Mask:
        """
        从硬盘加载 Mask 对象。
        """
        if not Path(self._mask_file_path).exists():
            raise FileNotFoundError(f"Mask file not found: {self._mask_file_path}")
        with open(self._mask_file_path, 'rb') as f:
            logger.debug(f"从磁盘加载 Mask: {self._mask_file_path}")
            return pickle.load(f)
    
    @property
    def data_type(self) -> DataType:
        return self._data_type
    
    def get_id_count_map(self) -> dict:
        return self.mask.get_id_count_map()
    
    def get_ids(self) -> List[int]:
        return list(self.mask.get_id_count_map().keys())


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