# coding: utf-8

import numpy as np

def box2coco(box):
    return [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]


def coco2box(coco):
    return [coco[0], coco[1], coco[0]+coco[2], coco[1]+coco[3]]