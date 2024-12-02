# coding: utf-8

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

from segment_anything import sam_model_registry, SamPredictor
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem
from schemas.process_result import Mask, ProcessResultItem, ProcessResult, Mask
from util.constant import CHECKPOINT_DIR
from util.logger import logger
from concurrent.futures import ProcessPoolExecutor
from typing import List


class SamProcessService():
    def __init__(self, ori_label_2_id_map: dict) -> None:
        self._ori_label_2_id_map = ori_label_2_id_map
        self._failed_items_file = "failed_items.txt"
        
    def generate_masks(self, data: PreprocessResult, use_gpu: False, parallel_num=0, merge_mask=True, limit=100, offset=0) -> List[Mask]:
        result = []
        num_gpus = 0
        if use_gpu:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs available, but 'use_gpu' is set to True")
            logger.info(f"Detected {num_gpus} GPUs")
            
        if parallel_num == 0: 
            if num_gpus > 0:
                parallel_num = num_gpus
            else:
                parallel_num = 1

            
        logger.info(f"Using {parallel_num} concurrent processes")
        
        l, r = offset, offset + limit
        if l >= len(data.result_list):
            return result
        
        if r > len(data.result_list):
            r = len(data.result_list)
        # 创建进程池
        try:
            with ProcessPoolExecutor(max_workers=parallel_num) as executor:
                futures = []
                for i, item in enumerate(data.result_list[l:r]):
                    try:
                        if use_gpu:
                            gpu_id = i % num_gpus
                            future = executor.submit(self._generate_masks_for_one_gpu, gpu_id, item, merge_mask=merge_mask)
                        else:
                            future = executor.submit(self._generate_masks_for_one_cpu, item, merge_mask=merge_mask)         
                        futures.append(future)
                    except Exception as e:
                        torch.cuda.empty_cache()  # 清理缓存
                        logger.error(f"Exception occurred: {e}. Retrying...")
                        # 记录失败的任务
                        with open(self._failed_items_file, 'a') as f:
                            f.write(f"{item.img_file_path}\n")

                # 等待所有任务完成
                for future in futures:
                    for ret in future.result():
                        result.append(ret)
                    logger.info(f"progress: {len(result)}/{len(data.result_list)}")
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt detected. Exiting.")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        return result
    
    
    def call_without_mask(self, data: PreprocessResult, merge_mask=True) -> ProcessResult:
        result = ProcessResult()
        for item in data.result_list:
            result_items = self.call_one_without_mask(item, merge_mask=merge_mask) 
            for result_item in result_items:
                result.append(result_item)
            
        return result


    # Deprecated
    def call(self, data: PreprocessResult, use_gpu: False, parallel_num=0, merge_mask=True) -> ProcessResult:
        result = ProcessResult()
        num_gpus = 0
        if use_gpu:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs available, but 'use_gpu' is set to True")
            logger.info(f"Detected {num_gpus} GPUs")
            
        if parallel_num == 0: 
            if num_gpus > 0:
                parallel_num = num_gpus
            else:
                parallel_num = 1

            
        logger.info(f"Using {parallel_num} concurrent processes")
        
        
        # 创建进程池
        try:
            with ProcessPoolExecutor(max_workers=parallel_num) as executor:
                futures = []
                for i, item in enumerate(data.result_list):
                    try:
                        if use_gpu:
                            gpu_id = i % num_gpus
                            future = executor.submit(self._call_on_gpu, gpu_id, item, merge_mask=merge_mask)
                        else:
                            future = executor.submit(self._call_on_cpu, item, merge_mask=merge_mask)         
                        futures.append(future)
                    except Exception as e:
                        torch.cuda.empty_cache()  # 清理缓存
                        logger.error(f"Exception occurred: {e}. Retrying...")
                        # 记录失败的任务
                        with open(self._failed_items_file, 'a') as f:
                            f.write(f"{item.img_file_path}\n")

                # 等待所有任务完成
                for future in futures:
                    for ret in future.result():
                        result.append(ret)
                    logger.info(f"progress: {len(result)}/{len(data.result_list)}")
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt detected. Exiting.")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        return result
    
    def _generate_masks_for_one_gpu(self, gpu_id, item, merge_mask=True):
        torch.cuda.set_device(gpu_id)
        return self.generate_masks_for_one(item, use_gpu=True, merge_mask=merge_mask)  # 调用你的单项处理逻辑
    
    def _generate_masks_for_one_cpu(self, item, merge_mask=True):
        return self.generate_masks_for_one(item, merge_mask=merge_mask)
    
    # Deprecated
    def _call_on_gpu(self, gpu_id, item, merge_mask=True):
        torch.cuda.set_device(gpu_id)
        return self.generate_masks_for_one(item, use_gpu=True, merge_mask=merge_mask)  # 调用你的单项处理逻辑
    
    # Deprecated
    def _call_on_cpu(self, item, merge_mask=True):
        return self.generate_masks_for_one(item, merge_mask=merge_mask)
    
    def generate_masks_for_one(self, item: PreprocessResultItem, use_gpu=False, merge_mask=True) -> List[Mask]:
        image = cv2.imread(item.img_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_checkpoint = os.path.join(CHECKPOINT_DIR, "sam_vit_h_4b8939.pth")
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        if use_gpu:
            sam.to(device="cuda")

        predictor = SamPredictor(sam)
        # slow
        predictor.set_image(image)

        ret = []
        mask = None
        for box_item in item.box_items:
            mask_arrs, scores = process_box_prompt(predictor, box_item.box_array)
            if mask_arrs is None or len(mask_arrs) == 0:
                raise Exception("No mask predicted")
            mask_arr = mask_arrs[0]
            score = scores[0]

            id = self._ori_label_2_id_map[box_item.ori_label]
            # 设置 box 对应的 id 和置信度值
            box_item.set_id(id)
            box_item.set_confidence_value(score)
            if merge_mask:
                if mask is None:
                    mask = Mask(item.img_file_path, mask_arr, id, box_items=[box_item])
                else:
                    mask.update(Mask(item.img_file_path, mask_arr, id, box_items=[box_item]))
                ret = [mask]
            else:
                mask = Mask(item.img_file_path, mask_arr, id, box_items=[box_item])
                ret.append(mask)
           
        return ret
    
    
    def call_one_without_mask(self, item: PreprocessResultItem, merge_mask=True) -> List[ProcessResultItem]:
        ret = []
        mask = None
        for box_item in item.box_items:
            id = self._ori_label_2_id_map[box_item.ori_label]
            if merge_mask:
                if mask is None:
                    mask = Mask(item.img_file_path, None, id, box_items=[box_item])
                else:
                    mask.update(Mask(item.img_file_path, None, id, box_items=[box_item]))
                ret = [ProcessResultItem(img_file_path=item.img_file_path, mask=mask, data_type=item.data_type)]
            else:
                ext = item.img_file_path.split("/")[-1].split(".")[1]
                mask_img_file_path = item.img_file_path.replace(f".{ext}", f"_{box_item.box_string()}_{box_item.ori_label}.{ext}")
                mask = Mask(item.img_file_path, None, id, box_items=[box_item])
                ret.append(ProcessResultItem(img_file_path=item.img_file_path, mask=mask, data_type=item.data_type,
                                             mask_img_file_path=mask_img_file_path))
           
        return ret
    

    def call_one(self, item: PreprocessResultItem, use_gpu=False, merge_mask=True) -> List[ProcessResultItem]: 
        image = cv2.imread(item.img_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_checkpoint = os.path.join(CHECKPOINT_DIR, "sam_vit_h_4b8939.pth")
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        if use_gpu:
            sam.to(device="cuda")

        predictor = SamPredictor(sam)
        # slow
        predictor.set_image(image)

        ret = []
        mask = None
        for box_item in item.box_items:
            mask_arrs, scores = process_box_prompt(predictor, box_item.box_array)
            if mask_arrs is None or len(mask_arrs) == 0:
                raise Exception("No mask predicted")
            mask_arr = mask_arrs[0]
            score = scores[0]

            id = self._ori_label_2_id_map[box_item.ori_label]
            # 设置 box 对应的 id 和置信度值
            box_item.set_id(id)
            box_item.set_confidence_value(score)
            if merge_mask:
                if mask is None:
                    mask = Mask(item.img_file_path, mask_arr, id, box_items=[box_item])
                else:
                    mask.update(Mask(item.img_file_path, mask_arr, id, box_items=[box_item]))
                ret = [ProcessResultItem(img_file_path=item.img_file_path, mask=mask, data_type=item.data_type)]
            else:
                ext = item.img_file_path.split("/")[-1].split(".")[1]
                mask_img_file_path = item.img_file_path.replace(f".{ext}", f"_{box_item.box_string()}_{box_item.ori_label}.{ext}")
                mask = Mask(item.img_file_path, mask_arr, id, box_items=[box_item])
                ret.append(ProcessResultItem(img_file_path=item.img_file_path, mask=mask, data_type=item.data_type,
                                             mask_img_file_path=mask_img_file_path))
           
        return ret


def process_box_prompt(predictor: SamPredictor, input_box: np.ndarray):
    logger.debug(f"processing box: {input_box}")
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    return masks, scores

