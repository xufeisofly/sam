# coding: utf-8
import os

from enum import Enum

ROOT_DIR = '/Users/sofly/projects/dataprocess/sam'
ORIGIN_DATA_DIR = os.path.join(ROOT_DIR, 'origin_data')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')


class DataType(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'