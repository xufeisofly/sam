# coding: utf-8
import os
import sys

import numpy as np
from torchvision.ops import box_area

from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType
import cv2


class SISIPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "SISI") if dataset_path is None else dataset_path
        self._image_path = os.path.join(self._dataset_path, "scenes/scenes")
        self._box_path = os.path.join(self._dataset_path, "shipsnet/shipsnet")

    def call(self, limit=-1) -> PreprocessResult:
        result = PreprocessResult()

        all_image_files = os.listdir(self._image_path)
        all_image_files = [f for f in all_image_files if f.endswith('.png')]
        all_box_files = os.listdir(self._box_path)
        all_box_files = [f for f in all_box_files if f.endswith('.png')]
        for image_file in all_image_files:
            img_file_path = os.path.join(self._image_path, image_file)
            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType.TRAIN)
            for box_file in all_box_files:
                box_file_path = os.path.join(self._box_path, box_file)
                area = self.locate(img_file_path, box_file_path)
                if area:
                    box_array = np.array(area)
                    result_item.append(BoxItem(
                        ori_label=self._get_ori_label_by_token(),
                        box_array=box_array))
            result.append(result_item)
        return result

    def ori_label_2_id_map(self) -> dict:
        return {
            'ship': 255,
        }

    def _get_ori_label_by_token(self) -> str:
        return 'ship'

    def test(self):
        all_image_files = os.listdir(self._image_path)
        all_image_files = [f for f in all_image_files if f.endswith('.png')]
        all_box_files = os.listdir(self._box_path)
        all_box_files = [f for f in all_box_files if f.endswith('.png')]
        for image_file in all_image_files:
            for box_file in all_box_files:
                area = self.locate(os.path.join(self._image_path, image_file), os.path.join(self._box_path, box_file))
                if area:
                    print("Match: ", image_file, box_file, area)
    def locate(self, image_file, box_file):
        # 读取大图和小图（模板）
        large_image = cv2.imread(image_file, 0)  # 0表示以灰度模式读取
        small_image = cv2.imread(box_file, 0)  # 模板也以灰度模式读取

        # 获取模板的宽高
        w, h = small_image.shape[::-1]

        # 使用匹配模板函数
        res = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)

        # 设定一个阈值，根据经验或实验确定
        threshold = 0.8

        # 获取匹配程度最高（或最低）的位置
        loc = np.where(res >= threshold)

        # 判断是否找到匹配的部分
        if len(loc[0]) > 0:
            # 获取匹配部分的左上角坐标（可能有多个匹配，这里只取第一个）
            top_left = loc[1][0], loc[0][0]
            bottom_right = top_left[0] + w, top_left[1] + h
            # # 输出匹配成功的信息
            # print("小图像是大图像的一部分。", image_file, box_file, top_left, bottom_right)
            #
            # # 在大图上绘制矩形框标出匹配部分
            # cv2.rectangle(large_image, top_left, bottom_right, 255, 2)
            #
            # # 显示结果
            # cv2.imshow('Matched', large_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        else:
            # 输出匹配失败的信息
            #print("小图像不是大图像的一部分。", image_file, box_file)
            return None

if __name__ == "__main__":
    service = SISIPreprocessService()
    service.test()
