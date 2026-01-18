# Dataset
You can download the dataset from the link: 
https://drive.google.com/file/d/1KLdH0i4bIJnzconPpPXWsMcmg06PcQSH/view

# Dependencies
Run it locally using python 3.11 and installing the dependencies:

Use uv:
```
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install -e py-vox-io
```

''
Alternative Conda:
conda env create -f environment.yml
''

# Activate the new environment:

Use uv:
```
source .venv/bin/activate
```

conda activate vasija
In this project, we use pyvox to open and write the vox files.
pyvox: https://github.com/gromgull/py-vox-io

# Resources:
We recommend you to complete the task on a GPU, maybe Google Colab 
or Baidu PaddleHub are some plausible options.

# Task 1 complete
填充修改了visualize.py里的函数，也添加了一些文件:
- 文件夹py-vox-io: 项目运行所需要的核心依赖库
- 文件夹test_visualize_data: 存放单元测试用的测试集和输出
- 文件test_visualize.ipynb: 单元测试的jupyter notebook, 按照他要求对函数进行了可视化测试。

# Task 2 complete

修改了 FragmentDataset.py ，现在可以通过

```
dataset = FragmentDataset('./data/train', '7')
dataset = FragmentDataset('./data/test', '7')
```

来分别获取训练和测试数据集。还可以通过

```
dataset.__get_item__(idx)
dataset.__get_item_specific_frag__(idx, select_frag)
```

分别来随机抽取一个碎片并返回，或是制定一个碎片 id 并返回

# Task 3 complete

添加了DCGAN和陶罐补全的原始论文，可供参考

补全了 utils/model.py，根据 handout 中的示意图实现了一个朴素GAN，其中：
- Generator的输入是 (cube_len ^ 3) 大小的0/1张量x，decoder的输出是同样大小的张量y，则最终输出（补全后的陶罐）为 x + (1 - x) * sigmoid(y)
- Discriminator的输入是 (resolution ^ 3) 大小的取值在[0, 1]内的张量，输出是[0, 1]内的概率，已经过sigmoid，如后续要改用BCEWithLogitsLoss需要去掉sigmoid

后续可改进的方向有：
- 加入skip connection
- 加入器型类别的embedding
- 修改网络架构/超参等
- ······

# Improvement 1

目前做了两处修改：
- 修改了utils/FragmentDataset.py中的部分函数，确保了输入值只有0或1（此前有可能是0或碎片编号）
- 在生成器的损失函数中增加了生成陶罐与真实陶罐之间的L1距离（与原始对抗损失的权重比为100:1），想法来自原论文，这一项能使训练更稳定，且让生成的结果更接近目标，而非仅仅看起来像个真实的陶罐；对抗损失能避免模糊，并完善细节。

在runs中保存了三个日志，分辨率均为32，分别是：
- original: 修改前的效果
- binary: 更正了FragmentDataset.py后的效果
- binaty+L1: 更正且增加了L1 Loss后的效果
初步证明了修改的有效性

两处修改后分辨率为64的程序还在运行中。

下一步计划：更改模型结构，加入skip connection。

# Improvement 2

更改了Generator的架构，增加了类似U-net的skip connection，想法来自原论文。
修改之后训练不太稳定，效果并未提高，之后把判别器的学习率缩小了10倍（相应增加了epoch数，目前看来可能没必要），64分辨率的训练暂时还未结束，但已经达到目前最好效果。

不确定model_utils.py中的generate函数是否存在问题（生成器的输出已经是完整陶罐，应该不需要再加上输入了？）

尝试了直接预测整个陶罐而非缺失部分，效果很差。

# Improvement 3

加了法向量，核心改变了FragmentDataset类中的相应函数和接口，现在getitem函数会返回voxel，法向量和表面mask。

在training.py中根据法向量和表面mask计算了相应损失，加在training.py中作为损失函数的监督信号，但是目前效果不佳。