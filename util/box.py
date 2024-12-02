# coding: utf-8

import numpy as np
from typing import Tuple

def box2coco(box):
    return [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]


def coco2box(coco):
    return [coco[0], coco[1], coco[0]+coco[2], coco[1]+coco[3]]


def poly2box(poly) -> np.ndarray:
    """取最小外接矩形
    """
    return np.array([int(min([p for p in poly[::2]])), int(min([p for p in poly[1::2]])), int(max([p for p in poly[::2]]))+1, int(max([p for p in poly[1::2]]))+1])


def calculate_island_area2(grid: np.ndarray, x, y):
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


def calculate_island_area(grid: np.ndarray, x, y):
    """
    计算给定位置 (x, y) 所在小岛的面积，不修改原始 grid。
    :param grid: 二维数组，元素是 0 或 1
    :param x: 起始行坐标
    :param y: 起始列坐标
    :return: 小岛的面积
    """
    if grid.size == 0 or grid[x, y] == 0:
        return 0

    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    value = grid[x, y]

    # 使用栈来模拟 DFS 过程
    stack = [(x, y)]  # 初始位置入栈
    visited[x, y] = True
    area = 0

    # 四个方向的移动
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        i, j = stack.pop()  # 弹出栈顶元素
        area += 1  # 当前点属于岛屿，面积加 1

        # 遍历四个方向
        for dx, dy in directions:
            ni, nj = i + dx, j + dy

            # 检查是否越界，是否已经访问过，或是否为岛屿
            if 0 <= ni < rows and 0 <= nj < cols and not visited[ni, nj] and grid[ni, nj] == value:
                visited[ni, nj] = True
                stack.append((ni, nj))  # 将有效的邻居坐标加入栈中

    return area



