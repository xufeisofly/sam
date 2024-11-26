# coding: utf-8
import os

from enum import Enum

ROOT_DIR = '/Users/sofly/projects/dataprocess/sam' if not os.environ.get('SAM_ROOT_DIR') else os.environ.get('SAM_ROOT_DIR')
ORIGIN_DATA_DIR = os.path.join(ROOT_DIR, 'origin_data')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')


class DataType(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'