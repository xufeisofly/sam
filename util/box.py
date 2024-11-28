# coding: utf-8

import numpy as np
from typing import Tuple

def box2coco(box):
    return [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]


def coco2box(coco):
    return [coco[0], coco[1], coco[0]+coco[2], coco[1]+coco[3]]


def calculate_island_area(grid: np.ndarray, x, y):
    """
    计算给定位置 (x, y) 所在小岛的面积。
    :param grid: 二维数组，元素是 0 或 1
    :param x: 起始行坐标
    :param y: 起始列坐标
    :return: 小岛的面积
    """
    # 检查边界条件
    value = grid[x][y]
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])

    # 定义递归的 DFS 方法
    def dfs(i, j):
        # 检查是否越界，是否已访问过（非 1）
        if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] != value:
            return 0

        # 将当前位置标记为已访问（避免重复计数）
        grid[i][j] = 0

        # 计算当前区域面积（1） + 四个方向的面积
        return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)

    # 开始计算小岛面积
    return dfs(x, y)