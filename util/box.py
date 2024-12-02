# coding: utf-8

import numpy as np
from typing import Tuple

def box2coco(box):
    return [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]


def coco2box(coco):
    return [coco[0], coco[1], coco[0]+coco[2], coco[1]+coco[3]]


def poly2box(poly):
    """取最小外接矩形
    """
    return [int(min([p for p in poly[::2]])), int(min([p for p in poly[1::2]])), int(max([p for p in poly[::2]]))+1, int(max([p for p in poly[1::2]]))+1]


def calculate_island_area(grid: np.ndarray, x, y):
    """
    计算给定位置 (x, y) 所在小岛的面积，不修改原始 grid。
    :param grid: 二维数组，元素是 0 或 1
    :param x: 起始行坐标
    :param y: 起始列坐标
    :return: 小岛的面积
    """
    # 检查边界条件
    value = grid[x][y]
    if not grid.any() or value == 0:
        return 0

    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)

    # 定义递归的 DFS 方法
    def dfs(i, j):
        # 检查是否越界，是否已访问过，或值不匹配
        if i < 0 or i >= rows or j < 0 or j >= cols or visited[i][j] or grid[i][j] != value:
            return 0

        # 标记当前位置为已访问
        visited[i][j] = True

        # 计算当前区域面积（1） + 四个方向的面积
        return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)

    # 开始计算小岛面积
    return dfs(x, y)
