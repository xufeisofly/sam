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
usage: sam [-h] [--dataset_path DATASET_PATH] [--use_gpu USE_GPU] [--gpu_ids GPU_IDS] [--parallel_num PARALLEL_NUM] [--limit LIMIT] [--merge_mask MERGE_MASK]
           [--chunk CHUNK] [--low_memory LOW_MEMORY] [--loglevel LOGLEVEL]
           dataset

positional arguments:
  dataset               数据集名称

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        指定数据集路径，不设置的话使用默认
  --use_gpu USE_GPU     是否使用 gpu
  --gpu_ids GPU_IDS     指定 gpu id eg 0,1,2,3 使用 list-gpu-ids 命令获取全部 gpu ids，仅 use_gpu=1 时生效
  --parallel_num PARALLEL_NUM
                        多进程数量，默认与 GPU 核数相同
  --limit LIMIT         图片处理数量 for train, val, test，默认处理所有
  --merge_mask MERGE_MASK
                        是否合并 mask 文件
  --chunk CHUNK         分批处理, -1=nochunk
  --low_memory LOW_MEMORY
                        低内存模式，会将部分大变量通过硬盘读取 0-默认模式，1-低内存模式
  --loglevel LOGLEVEL   Set the log level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
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
chunk 参数可以分批次运行 sam 处理，默认为 100，当图片很多时可以节省内存，如果设置为 -1 时会一次性读到内存，然后进行处理
```
sam MAR20 --chunk=-1
```

设置了 chunk 后运行一旦出错，会在 {dataset}_fail.json 中记录出错的 chunk 断点，方便修复错误后继续运行

# 运行样例

不对 mask 进行合并（跑样例看效果时使用）
```
sam MAR20 --merge_mask=0
```

Notice: 样例不会对单张图片的 masks 进行合并，如果 masks 过多会导致内存爆炸，为此可使用 low_memory 模式，该模式下 mask 不分配内存，而是通过磁盘进行读写临时文件。代码会在一张图片的 masks 总大小超过 512MB 时自动开启 low_memory

```
# 此情况建议按如下参数运行
sam MAR20 --merge_mask=0 --low_memory=1 --chunk=1
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