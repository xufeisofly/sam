# coding: utf-8
import PIL
import argparse
import matplotlib.pyplot as plt
import cv2
import sys
import json

from util.box import coco2box

def show_box():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help='图片路径', default=None, type=str)
    parser.add_argument('--bbox', help='xmin, ymin, width, height, eg. 100,200,10,20', default=None, type=str)
    parser.add_argument('--detection_file', help='detection_data/annotations 中的 json 文件', default=None, type=str)
    args = parser.parse_args()
    
    boxs = []
    if args.detection_file is not None:
        with open(args.detection_file, 'r') as file:
            data = json.load(file)
        img_name = args.img_path.split('/')[-1] 
        annos = data['annotations']
        for anno in annos:
            if anno['image'].endswith(img_name):
                boxs.append(coco2box(anno['bbox']))
    else:
        args.bbox = [int(i) for i in args.bbox.split(',')]
        boxs = [coco2box(args.bbox)]         

    
    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    for box in boxs:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
    
    plt.axis('off')
    plt.show()
    
    sys.exit(0)