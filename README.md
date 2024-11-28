# Install

```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
# 调试模式
pip install -e .
```

# Run

```
usage: sam [-h] [--dataset_path DATASET_PATH] [--use_gpu USE_GPU] [--parallel_num PARALLEL_NUM] [--limit LIMIT] [--merge_mask MERGE_MASK]
           dataset

positional arguments:
  dataset               数据集名称

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        指定数据集路径，不设置的话使用默认
  --use_gpu USE_GPU     是否使用 gpu
  --parallel_num PARALLEL_NUM
                        多进程数量
  --limit LIMIT         图片处理数量 for train, val, test，默认处理所有
  --merge_mask MERGE_MASK
                        是否合并 mask 文件
```

正式运行
```
sam MAR20

```
# 参数

指定原始数据集目录(需要 preprocessor 对接)
```
sam MAR20 --dataset_path /data/mar20
```

只处理前 100 张图片（需要 preprocessor 对接）
```
sam MAR20 --limit 100
```

# 预查看分割效果

不对 mask 进行合并（跑样例看效果时使用）
```
sam MAR20 --merge_mask=0
```

show-box 可以展示图片中框的位置，用于查看分割效果

```
(venv) sam [main●] % show-box -h
usage: show-box [-h] [--bbox BBOX] [--detection_file DETECTION_FILE] img_path

positional arguments:
  img_path              图片路径

options:
  -h, --help            show this help message and exit
  --bbox BBOX           xmin, ymin, width, height, eg. 100,200,10,20
  --detection_file DETECTION_FILE
                        detection_data/annotations 中的 json 文件
```

展示 detection_data/annotations json 文件中所有 bbox 的位置
```
show-box '/Users/sofly/projects/dataprocess/sam/output/SODA-D/img_dir/test/00065.tif' --detection_file=/Users/sofly/projects/dataprocess/sam/output/SODA-D/detection_data/annotations/test.json
```

单独展示一个 bbox 的位置
```
show-box '/Users/sofly/projects/dataprocess/sam/output/SODA-D/img_dir/test/00065.tif' --bbox=861,1770,23,50
```