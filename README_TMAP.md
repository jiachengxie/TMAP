
# 基于轨迹遍历与混合注意力的车辆轨迹预测

![](https://github.com/nachiket92/PGP/blob/main/assets/intro.gif)




该仓库包含"基于轨迹遍历与混合注意力的车辆轨迹预测"的代码,作者是:谢佳成,雷俊杰,金朋帅, 为第五届中国研究生人工智能创新大赛的提交作品.

```bibtex
@inproceedings{deo2021multimodal,
  title={基于轨迹遍历与混合注意力的车辆轨迹预测},
  author={[谢佳成,雷俊杰,金朋帅]},
  booktitle={第五届中国研究生人工智能创新大赛},
  year={2023}
}
```


**Note:** 代码的实现请参照以下的步骤,希望对您有所帮助


## Installation

1. 从github上克隆该代码

**Note:** 我们使用的是Ubuntu-20.04版本,当前的Windows由于多线程能力有问题,运行该代码会导致ray报错
2. 创建一个新的conda环境(以免各个库版本冲突) 
``` shell
conda create --name TMAP python=3.7
```

3. 下载依赖(如果运行时还缺少什么就下载什么)
```shell
conda activate TMAP

# nuScenes devkit
pip install nuscenes-devkit

# 所用版本为Pytorch 1.7.1, CUDA 10.1, 可以更新,根据自己的实际情况来选择
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.0.221 -c pytorch

# 其他库
pip install ray
pip install psutil
pip install positional-encodings==5.0.0
pip install imageio
pip install tensorboard
```


## Dataset

1. 下载 [nuScenes dataset](https://www.nuscenes.org/download). 为实现该项目,下载这两个部分.
    - Metadata for the Trainval split (v1.0)
    - Map expansion pack (v1.3)

2. 在nuScenes的根目录中排列如下
```plain
└── nuScenes/
    ├── maps/
    |   ├── basemaps/
    |   ├── expansion/
    |   ├── prediction/
    |   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
    |   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
    |   ├── 53992ee3023e5494b90c316c183be829.png
    |   └── 93406b464a165eaba6d9de76ca09f5da.png
    └── v1.0-trainval
        ├── attribute.json
        ├── calibrated_sensor.json
        ...
        └── visibility.json         
```

3. 可以直接在cmd中输入以下代码,进行数据的预处理,以执行后续的训练,验证和可视化
```shell
python preprocess.py -c configs/preprocess_nuscenes.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data
```


## Inference

已经训练好的模型参数放在项目目录中的OUTCOME中的checkpoints中了,自己看results里的文本文件中展示的性能,选择所需要的权重.

可以直接在cmd中输入以下代码,以直接验证,会输出一个记录性能的文本文件.
```shell
python evaluate.py -c configs/TMAP_configs.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -w path/to/trained/weights
```

可以直接在cmd中输入以下代码,以直接将预测结果可视化,会输出一系列gif图,注意,如果内存不足会直接停止.
```shell
python visualize.py -c configs/TMAP_configs.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -w path/to/trained/weights
```


## Training

在cmd中输入以下代码,进行模型训练
```shell
python train.py -c configs/TMAP_configs.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -n 100
```

每一轮训练结束后会在指定目录中保存训练的权重和日志

启动tensorboard, 运行:
```shell
tensorboard --logdir=path/to/output/directory/tensorboard_logs
```
