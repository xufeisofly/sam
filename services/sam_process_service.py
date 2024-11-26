# coding: utf-8

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

from segment_anything import sam_model_registry, SamPredictor
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem
from schemas.process_result import Mask, ProcessResultItem, ProcessResult
from util.constant import CHECKPOINT_DIR
from util.logger import logger
from concurrent.futures import ProcessPoolExecutor


class SamProcessService():
    def __init__(self, ori_label_2_id_map: dict) -> None:
        self._ori_label_2_id_map = ori_label_2_id_map

    def call(self, data: PreprocessResult, use_gpu: False, parallel_num=1) -> ProcessResult:
        result = ProcessResult()
        if use_gpu:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs available, but 'use_gpu' is set to True")
            logger.info(f"Detected {num_gpus} GPUs")
        logger.info(f"Using {parallel_num} concurrent processes")
        
        # 创建进程池
        with ProcessPoolExecutor(max_workers=parallel_num) as executor:
            futures = []
            for i, item in enumerate(data.result_list):
                if use_gpu:
                    gpu_id = i % num_gpus
                    futures.append(executor.submit(self._call_on_gpu, gpu_id, item))
                else:
                    futures.append(executor.submit(self._call_on_cpu, item))

            # 等待所有任务完成
            for future in futures:
                result.append(future.result())
                logger.info(f"progress: {len(result)}/{len(data.result_list)}")

        return result
    
    def _call_on_gpu(self, gpu_id, item):
        torch.cuda.set_device(gpu_id)
        return self.call_one(item, use_gpu=True)  # 调用你的单项处理逻辑
    
    def _call_on_cpu(self, item):
        return self.call_one(item)


    def call_one(self, item: PreprocessResultItem, use_gpu=False) -> ProcessResultItem: 
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

        mask = None
        for box_item in item.box_items:
            mask_arrs, _ = process_box_prompt(predictor, box_item.box_array)
            if mask_arrs is None or len(mask_arrs) == 0:
                raise Exception("No mask predicted")
            mask_arr = mask_arrs[0]
            id = self._ori_label_2_id_map[box_item.ori_label]
            # 设置 box 对应的 id
            box_item.set_id(id)
            if mask is None:
                mask = Mask(item.img_file_path, mask_arr, id, box_items=[box_item])
            else:
                mask.update(Mask(item.img_file_path, mask_arr, id, box_items=[box_item]))

        return ProcessResultItem(img_file_path=item.img_file_path, mask=mask)


def process_box_prompt(predictor: SamPredictor, input_box: np.array):
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    return masks, scores

