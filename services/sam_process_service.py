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
from concurrent.futures import ProcessPoolExecutor


class SamProcessService():
    def __init__(self, ori_label_2_id_map: dict) -> None:
        self._ori_label_2_id_map = ori_label_2_id_map

    def call(self, data: PreprocessResult, use_gpu: False, parallel_num=1) -> ProcessResult:
        result = ProcessResult()
        print(f"Using {parallel_num} concurrent")
        if use_gpu:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs available, but 'use_gpu' is set to True")
            print(f"Detected {num_gpus} GPUs")
        print(f"Using {parallel_num} concurrent processes")
        
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
            print(id, box_item.ori_label)
            if mask is None:
                mask = Mask(item.img_file_path, mask_arr, id, box_items=[box_item])
            else:
                mask.update(Mask(item.img_file_path, mask_arr, id, box_items=[box_item]))

        return ProcessResultItem(img_file_path=item.img_file_path, mask=mask)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_image_with_masks(image, masks, scores):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=figsize(image), dpi=100)
        plt.imshow(image)
        show_mask(mask, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()      

def process_points_prompt(predictor: SamPredictor, points: np.array, labels: np.array):
    masks, _, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_points(points, labels, plt.gca())
    plt.axis('off')
    plt.show()

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True)
    
    return masks, scores

def process_box_prompt(predictor: SamPredictor, input_box: np.array):
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    return masks, scores

def figsize(image, dpi=100):
    height, width, _ = image.shape
    figsize = width / dpi, height / dpi
    return figsize

image_path1 = 'images/truck.jpg'
image_path2 = '../MARS20/JPEGImages/JPEGImages/1.jpg'

if __name__ == '__main__':
    image = cv2.imread(image_path2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "../checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    #sam.to(device="cuda")

    predictor = SamPredictor(sam)
    # slow
    predictor.set_image(image)

    # input_point = np.array([[500, 375]])
    # input_label = np.array([1])
    # points_prompt(predictor, input_point, input_label)

    # left, top, right, bottom
    input_boxes = np.array([[485, 427, 554, 500],
                            [694, 487, 770, 562],
                            [58, 205, 134, 285]])
    

    plt.figure(figsize=figsize(image), dpi=100)
    plt.imshow(image)
    for input_box in input_boxes:
        masks, scores = process_box_prompt(predictor, input_box)
        show_box(input_box, plt.gca())
        show_mask(masks[0], plt.gca())
    plt.axis('off')
    plt.show()