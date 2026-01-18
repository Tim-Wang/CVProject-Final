# 数据集
在

https://drive.google.com/file/d/1KLdH0i4bIJnzconPpPXWsMcmg06PcQSH/view

下载数据集后，解压至 ./data 目录。

# 依赖
请使用 uv 安装依赖。运行下面的命令：
```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install -e py-vox-io
```

# 训练

先激活环境

```bash
source .venv/bin/activate
```

然后，直接运行训练代码

```bash
python training.py
```

训练时，会向 ./runs/ 写训练日志，可用命令

```bash
tensorboard --logdir ./runs/
```
监看。

训练时还会向 ./checkpoints/ 写入权重。

# 可视化训练效果

先从

https://disk.pku.edu.cn/link/AAEB65273F86D243C394DA91C796785D5F

下载 checkpoint 。

打开 visualize_train_result.ipynb ，选择正确的 kernel 后，填入 checkpoint 路径。然后，依次运行各 block ，可以看到模型在某特定碎片上的补全结果。

