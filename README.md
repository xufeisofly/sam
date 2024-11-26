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
llm-round3002 :: ~/dataprocess/sam ‹main*› » sam --help
usage: sam [-h] [--use_gpu USE_GPU] [--parallel_num PARALLEL_NUM] dataset

positional arguments:
  dataset               数据集名称

options:
  -h, --help            show this help message and exit
  --use_gpu USE_GPU     是否使用 gpu
  --parallel_num PARALLEL_NUM
                        多进程数量
```

```
sam MAR20 --use_gpu=1 --parallel_num=2
or
python sam/main.py MAR20 --use_gpu=0 --parallel_num=2
```
