# 1 Pytorch 环境配置

## Anaconda 安装

## 显卡配置(驱动+CUDA Toolkit)

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201237544.png" alt="image-20230718205108412" style="zoom: 33%;" />

## 有序地管理环境

初始环境：base

切换环境使用不同的pytorch版本

```c
//输入命令，安装python
conda create -n pytorch python=3.7
    //输入命令，激活环境
    conda activate pytorch
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238878.png" alt="image-20230718210634806" style="zoom:67%;" />

```c
//查看工具包
pip list
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238876.png" alt="image-20230718210727247" style="zoom: 67%;" />

## Pytorch安装

官网：https://pytorch.org/ 

任务管理器查看是否有英伟达显卡

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238020.png" alt="image-20230718212126175" style="zoom: 33%;" />

CUDA推荐使用9.2

查看驱动版本

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238219.png" alt="image-20230718212421748" style="zoom: 50%;" />

大于396.26可使用

pytorch环境下输入命令，安装9.2版本

```
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults
c numba/label/dev
```

![image-20230718213642078](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238093.png)

报错，因为下载速度太慢

![image-20230718215019872](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238193.png)

清华源可以下载cpu版本：https://blog.csdn.net/zzq060143/article/details/88042075

如果找不到源，需要把命令中的 https 改成 http

下载gpu版本教程：https://www.bilibili.com/read/cv15186754

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238461.png" alt="image-20230718233907433" style="zoom:67%;" />

返回时False，因为装的是cpu版本，gpu版本才返回true。cpu版本学习阶段可以使用。

![image-20230719121436991](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238202.png)

# 2 Python编辑器的选择

## Pytorch安装

官网：https://www.jetbrains.com/pycharm/

下载Community版本

## Pytorch 配置

create new project

需要自己配置解释器

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238045.png" alt="image-20230719100104852" style="zoom:50%;" />

添加python.exe

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201238038.png" alt="image-20230719103201967" style="zoom:50%;" />

Conda Environment可能找不到python.exe，选择System Environment添加

https://blog.csdn.net/weixin_43537097/article/details/130931535

打开Python Consle

import torch

输入torch.cuda.is_available()，CPU版本返回false

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201239602.png" alt="image-20230719103754349" style="zoom: 50%;" />

右侧工具栏可实时查看变量

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201239825.png" alt="image-20230719104030691" style="zoom:50%;" />

## Jupyter 安装

在Pytorch环境中安装Jupyter

在pytorch环境中安装一个包

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201239084.png" alt="image-20230719131043395" style="zoom:67%;" />

运行Jupyter

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201239811.png" alt="image-20230719131433335" style="zoom: 50%;" />

创建代码

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201239442.png" alt="image-20230719131509833" style="zoom:50%;" />

shift + enter运行代码块

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201239418.png" alt="image-20230719131558361" style="zoom:50%;" />

# 3 Pytorch学习中的两大法宝函数 

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201254546.png" alt="image-20230719140047441" style="zoom:50%;" />   

**总结：**
	dir()函数，能让我们知道工具箱以及工具箱中的分隔区有什么东西。
	help()函数，能让我们知道每个工具是如何使用的，工具的使用方法。

## 打开Pycharm，测试这两个工具函数

```py
dir(torch.cuda.is_available)
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201254923.png" alt="image-20230719141153933" style="zoom:50%;" />

前后有双下划线，表明变量不能修改，说明是函数，不是分割区

dir和help里面函数后面的括号记得去掉

```py
help(torch.cuda.is_available)
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201254498.png" alt="image-20230719141419656" style="zoom:67%;" />

# 4 Pycahrm及Jupyter使用对比

## 在Pycharm中新建项目

### 在File-Setting中可查看该项目是否有Pytorch环境

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201254410.png" alt="image-20230719141734242" style="zoom:50%;" />

### 新建Python文件

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201254822.png" alt="image-20230719141855063" style="zoom:50%;" />

### 为Python文件设置Python解释器

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255941.png" alt="image-20230719142138096" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255929.png" alt="image-20230719142305822" style="zoom:50%;" />

运行成功

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255878.png" alt="image-20230719142342738" style="zoom:50%;" />

也可以直接在Python控制台输入语句，直接输出结果

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255536.png" alt="image-20230719142538947" style="zoom:50%;" />

## Jupyter新建项目及使用

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255848.png" alt="image-20230719142709123" style="zoom:50%;" />

## 三种代码编辑方式对比

用三种方式运行同一段错误代码

### Python文件

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255869.png" alt="image-20230719142919805" style="zoom: 50%;" />

报错，字符串和整型相加不允许

修改b后，运行成功

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255241.png" alt="image-20230719143050346" style="zoom:50%;" />

### Python控制台

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255322.png" alt="image-20230719143236205" style="zoom:50%;" />

修改b后

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255432.png" alt="image-20230719143314888" style="zoom:67%;" />

如果发生错误，代码可读性下降

shift+enter可以以多行为一个块运行

![image-20230719143806537](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255933.png)

### Jupyter

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255001.png" alt="image-20230719143423641" style="zoom:50%;" />

修改b后

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255789.png" alt="image-20230719143509947" style="zoom:67%;" />

### ·总结

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255357.png" alt="image-20230719144020108" style="zoom:50%;" />

# 5 Pytorch加载数据初认识

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201255707.png" alt="image-20230719145042656" style="zoom:50%;" />

## 下载蚂蚁/蜜蜂数据集

## 创建read_data.py文件

```python
from torch.utils.data import Dataset
```

Jupyter中可查看Dateset内的函数

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201256186.png" alt="image-20230719150017028" style="zoom:50%;" />

# 6 Dataset类代码实战

第一次打开终端报错解决：https://blog.csdn.net/qq_33405617/article/details/119894883

## 导入Image

```python
from PIL import Image
```

将 “蚂蚁/蜜蜂” 数据集复制到项目中

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201256929.png" alt="image-20230719153011629" style="zoom: 50%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201256692.png" alt="image-20230719153152359" style="zoom:67%;" />

## Python控制台中读取数据

```python
from PIL import Image
```

复制图片绝对路径，\改成\\表示转义

```python
img_path = "D:\\PytorchLearning\\dataset\\train\\ants\\0013035.jpg"
```

![image-20230719153920417](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201256227.png)

```python
# 读取图片
img = Image.open(img_path)
```

![image-20230719154131158](C:/Users/ge'yu/AppData/Roaming/Typora/typora-user-images/image-20230719154131158.png)

```python
img.size
# Out[5]: (768, 512)
```

```python
# 查看图片
img.show()
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201256350.png" alt="image-20230719154934779" style="zoom:50%;" />

## 获取图片名称及路径

### 控制台方式

```python
# ants文件夹相对路径
dir_path = "dataset/train/ants"
# 导入os
import os
# 将ants文件夹下的图片生成列表
img_path_list = os.listdir(dir_path)
# 获取第一张图片
img_path_list[0]
# Out[10]: '0013035.jpg'
```

### python文件方式

```python
# python文件中.
# 每张图片的label就是所在的文件夹的名称
def __init__(self, root_dir, label_dir):
# 控制台
import os
root_dir = "dataset/train"
label_dir = "ants"
# 拼接路径
path = os.path.join(root_dir, label_dir)

# pyhton文件修改init函数
# 函数之间的参数不能相互使用，但是self制定了一个类中的全局变量，相当于c++的static
 def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 所有图片名称列表
        self.img_path = os.listdir(self.path)
        
 # 控制台中
img_path = os.listdir(path)

# 获取每一个图片
# 修改 —__getitem__函数
  def __getitem__(self, idx):
        # 图片名
        img_name = self.img_path[idx]
        # 每个图片的相对路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 根据图片路径读取图片
         img = Image.open(img_item_path)
        # 图片的标签
         label = self.label_dir
         return img, label

 # 控制台检验
idx = 0
# 注意这里是中括号
img_name = img_path[idx]
img_item_path = os.path.join(root_dir, label_dir, img_name)
 img = Image.open(img_item_path)
```

### 数据集长度

```python
 def __len__(self):
        return len(self.img.path)
```

### 创建实例

```python
# 创建实例
root_dir = "dataset/train"
ants_label_dir = "ants"
ants_dataset = MyData(root_dir, ants_label_dir)
```

### 控制台运行

对象中包含init中的所有变量

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201256144.png" alt="image-20230719162922969" style="zoom:67%;" />

```python
ants_dataset[0]
'''
Out[5]: (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512>, 'ants')
这段代码表示访问了一个名为"ants_dataset"的数据集中的第一个数据项。该数据项包含一张图片和一个标签。
图片格式为JPEG，具体尺寸为768x512像素，采用RGB颜色模式。标签为"ants"，表示这张图片中的内容是蚂蚁（ants）。
'''
# 分别获取图片和标签
img, label = ants_dataset[0]
```

### 同时有蚂蚁和蜜蜂数据集

```python
# 创建实例
root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
```

### 两个数据集集合

```python
train_dataset = ants_dataset + bees_dataset
```

## txt标签方式

修改数据集文件名，添加标签文件夹

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201256435.png" alt="image-20230719165632515" style="zoom:67%;" />

## 添加标签

标签txt的名称与图片名称一致，txt内容为标签值

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day1202307201256609.png" alt="image-20230719165752353" style="zoom:50%;" />

# 7 Tensorboard的使用(一)

## 打开Pycharm，设置环境

```python
# 从torch.utils.tensorboard模块中导入SummaryWriter类
from torch.utils.tensorboard import SummaryWriter
# 将事件和文件存储到"logs"文件夹下
writer = SummaryWriter("logs")

# writer.add_image()
# y = x
# i 范围是0到99
for i in range(100):
    writer.add_scalar("y = x", i, i)

write.close()
```

## add_scalar()方法

![image-20230720095510726](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201726884.png)

```python
 def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
        """Add scalar data to summary.
        Args:
            tag (str): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
        """
```

## 安装TensorBoard

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201726409.png" alt="image-20230720100426926" style="zoom:67%;" />

安装后再次运行，左侧多了一个logs文件

![image-20230720100707696](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201726488.png)

终端输入

```python
tensorboard --logdir=logs
'''
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.11.2 at http://localhost:6006/ (Press CTRL+C to quit)
'''
```

指定端口

```python
tensorboard --logdir=logs --port=6007
```

访问端口，显示图像

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201726825.png" alt="image-20230720101433132" style="zoom:50%;" />

绘制y=2x

```python
for i in range(100):
    writer.add_scalar("y = 2x", 2*i, i)
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201726701.png" alt="image-20230720101628747" style="zoom:50%;" />

如果不改变add_scalar()函数的标题只改变参数

```python
for i in range(100):
    writer.add_scalar("y = 2x", 3*i, i)
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201726734.png" alt="image-20230720101754325" style="zoom:50%;" />

![image-20230720101845019](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201726357.png)

向writer中写入新的事件，同时也记录了上一个事件

### 解决方法：

一、删除logs下的文件，重新启动程序

二、创建子文件夹，也就是说创建新的SummaryWriter("新文件夹")

# 8 TensorBoard的使用（二）add image()的使用(常用来观察训练结果)

控制台输入

```python
image_path = "data/train/ants_image/0013035.jpg"
from PIL import Image
# 读取图片
img = Image.open(image_path)
# 查看类型
print(type(img))
# <class PIL.JpegImagePlugin.JpegImageFile'>
```

## 利用numpy.array()，对PIL图片进行转换

NumPy型图片是指使用NumPy库表示和处理的图像。NumPy是一个广泛使用的Python库，用于科学计算和数据处理。它提供了一个多维数组对象（ndarray），可以用于存储和操作大量的数值数据。在图像处理领域中，NumPy数组通常用来表示图像的像素值。

NumPy数组可以是一维的（灰度图像）或二维的（彩色图像）。对于彩色图像，通常使用三维的NumPy数组表示，其中第一个维度表示图像的行数，第二个维度表示图像的列数，第三个维度表示图像的通道数（例如，红、绿、蓝通道）

### 控制台

```python
import numpy as np
img_array = np.array(img)
print(type(img_array))
# <class 'numpy.ndarray'>
```

### 文件内

```python
# 从torch.utils.tensorboard模块中导入SummaryWriter类
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 将事件和文件存储到"logs"文件夹下
writer = SummaryWriter("logs")
image_path = "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
# (H,W,C)——(高度，宽度，通道)
print(img_array.shape)
# 需要指定格式
writer.add_image("test", img_array, 1, dataformats='HWC' )
```

从PIL到numpy, 需要在add image()中指定shape中每一个数字/维表示的含义。

打开端口，显示图像

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201726463.png" alt="image-20230720104855434" style="zoom:50%;" />

添加蜜蜂图片，修改步长为2

```python
image_path = "data/train/bees_image/16838648_415acd9e3f.jpg"
writer.add_image("test", img_array, 2, dataformats='HWC' )
```

<img src="C:/Users/ge'yu/AppData/Roaming/Typora/typora-user-images/image-20230720105220800.png" alt="image-20230720105220800" style="zoom:50%;" />

更换标题

```python
writer.add_image("train", img_array, 1, dataformats='HWC' )
```

![image-20230720105351794](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning note/Day2202307201726332.png)

# 9 Transforms 的使用（一）

## transforms结构及用法

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727532.png" alt="image-20230720112701203" style="zoom:50%;" />

ctrl+p可提示函数需要什么参数

```python
from PIL import Image
from  torchvision import transforms

# 通过transform.ToTensor去解决两个问题
# 1、transform该如何使用
# 2、为什么需要Tensor这种数据类型

img_path = "train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)
# 输出：<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x1DCFAD7CFC8>
# 创建Totensor对象
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)
'''
输出：
tensor([[[0.3137, 0.3137, 0.3137,  ..., 0.3176, 0.3098, 0.2980],
         [0.3176, 0.3176, 0.3176,  ..., 0.3176, 0.3098, 0.2980],
         [0.3216, 0.3216, 0.3216,  ..., 0.3137, 0.3098, 0.3020],
         ...,
         [0.3412, 0.3412, 0.3373,  ..., 0.1725, 0.3725, 0.3529],
         [0.3412, 0.3412, 0.3373,  ..., 0.3294, 0.3529, 0.3294],
         [0.3412, 0.3412, 0.3373,  ..., 0.3098, 0.3059, 0.3294]],

        [[0.5922, 0.5922, 0.5922,  ..., 0.5961, 0.5882, 0.5765],
         [0.5961, 0.5961, 0.5961,  ..., 0.5961, 0.5882, 0.5765],
         [0.6000, 0.6000, 0.6000,  ..., 0.5922, 0.5882, 0.5804],
         ...,
         [0.6275, 0.6275, 0.6235,  ..., 0.3608, 0.6196, 0.6157],
         [0.6275, 0.6275, 0.6235,  ..., 0.5765, 0.6275, 0.5961],
         [0.6275, 0.6275, 0.6235,  ..., 0.6275, 0.6235, 0.6314]],

        [[0.9137, 0.9137, 0.9137,  ..., 0.9176, 0.9098, 0.8980],
         [0.9176, 0.9176, 0.9176,  ..., 0.9176, 0.9098, 0.8980],
         [0.9216, 0.9216, 0.9216,  ..., 0.9137, 0.9098, 0.9020],
         ...,
         [0.9294, 0.9294, 0.9255,  ..., 0.5529, 0.9216, 0.8941],
         [0.9294, 0.9294, 0.9255,  ..., 0.8863, 1.0000, 0.9137],
         [0.9294, 0.9294, 0.9255,  ..., 0.9490, 0.9804, 0.9137]]])
'''
```

# 10Transforms的使用（二）

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727314.png" alt="image-20230720113024746" style="zoom:50%;" />

Tensor包括深度学习需要的参数

## 下载Opencv

终端输入

```python
pip install opencv-python
```

控制台

```python
import cv2
cv_img = cv2.imread(img_path)
```

## 利用Tensor_img显示图片

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from  torchvision import transforms

img_path = "train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
# 输出：<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x1DCFAD7CFC8>
# 创建Totensor对象
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
```

终端输入

```python
tensorboard --logdir=logs
```

打开端口，显示图片

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727795.png" alt="image-20230720115601972" style="zoom:67%;" />

# 11 常见的Transforms（一）

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727366.png" alt="image-20230720141126141" style="zoom:67%;" />

## Pytorch中call()的用法

```python
class Person:
    def __call__(self, name):
        print("__call__" + "hello" + name)

    def hello(self, name):
        print("hello"+name)

person = Person()
person("zhangsan")
person.hello("list")
# __call__可以采用对象(参数)的方式调用，不用加.方法名
```

## ToTensor的使用

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("train/ants_image/0013035.jpg")
print(img)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor", img_tensor)
writer.close()
```

## Normalize() 归一化 的使用

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727404.png" alt="image-20230720143339908" style="zoom:67%;" />

mean是均值，std是标准差

```python
# 每个通道都有均值和标准差
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print((img_norm[0][0][0]))
'''
输出
tensor(0.3137)
tensor(-0.3725)
0.3137*2-1=-0.3725(四舍五入后结果)
'''
```

输出归一化结果

```python
writer.add_image("Normalize", img_norm)
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727313.png" alt="image-20230720143837709" style="zoom:67%;" />

# 12 常见的Transforms（二）

## Resize()的使用

```python
print(img.size)
# Resize有两个括号(h,w)
trans_resize = transforms.Resize((480, 480))
img_resize = trans_resize(img)
print(img_resize)
'''
输出
(768, 512)
<PIL.Image.Image image mode=RGB size=480x480 at 0x1C6224EBE88>'''
```

将PIL类型的img_resize转为tensor类型

```python
print(img.size)
# Resize有两个括号(h,w)
trans_resize = transforms.Resize((480, 480))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resieze PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)
```

图片大小改变

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727271.png" alt="image-20230720150314504" style="zoom:67%;" />

## Compose()的使用

将不同的操作组合起来，按顺序执行。前一步的输出是下一步的输入，要对应。

Compose()中的参数需要是一个列表。Python中，列表的表示形式为[数据1，数据2,...]。在Composel中，数据需要是transforms类型，所以得到，Compose([transforms参数1，transforms参数2,...])

## RandomCrop()随机裁剪的用法

```python
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727464.png" alt="image-20230720152147175" style="zoom:67%;" />

## 总结使用方法

- 关注输入和输出类型
- 多看官方文档
- 关注方法需要什么参数
- 不知道返回值的时候
  - print()
  - print(type())
  - debug

# 13 torchvision中的数据集使用

## 下载训练集和测试集

```python
import torchvision

# root表示数据集路径;train为true表示训练集,false表示测试集;download为true会自动从官网下载
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, dowmload=True)

```

可以用迅雷加快下载速度

```python
print(test_set[0])
# 表示有img 和 target两个属性
# 输出(<PIL.Image.Image image mode=RGB size=32x32 at 0x22BF14AA548>, 3)
```

classes内表示每种target对应哪种类别

```python
print(test_set.classes)
img, target = test_set[0]
print(img)
print(target)
# target对应的类型
print(test_set.classes[target])
# 显示图片
img.show()
'''
输出
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
<PIL.Image.Image image mode=RGB size=32x32 at 0x2622292ABC8>
3
cat
'''
```

## 添加Transform参数

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()

])
# root表示数据集路径;train为true表示训练集,false表示测试集;download为true会自动从官网下载
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

print(test_set[0])
writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201727933.png" alt="image-20230720161746404" style="zoom:67%;" />

# 14 DataLoader的使用

## 测试数据集中第一张图片及target

```python
import  torchvision
from torch.utils.data import DataLoader

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)
'''
torch.Size([3, 32, 32])
3
'''
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201728188.png" alt="image-20230720164801159" style="zoom:67%;" />

## 理解batch_size

```python
for data in test_loader:
    imgs, targets = data
    print(imgs,shape)
    print(targets)
 '''
 # 每次循环取4张图片，每张图片3个通道，32*32
 torch.Size([4, 3, 32, 32])
 # 每张图片的target
tensor([0, 3, 0, 2])
 '''
```

### 更改batch_size=64

```python
writer = SummaryWriter("dataloader")
step = 0
# 每次取64张图片
for data in test_loader:
    imgs, targets = data
    # 注意是add_images
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()
```



<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201728884.png" alt="image-20230720171254181" style="zoom:67%;" />

drop_last设置为false，所以不会丢掉数量小于batch_seze的组。

## 理解shuffle

添加epoch

```python
# shuffle设置为false
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

for epoch in range(2):
    writer = SummaryWriter("dataloader")
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()
'''
"Epoch: {}".format(epoch)是一种字符串格式化的方法，在Python中常用于将变量的值插入到字符串中的特定位置。
在这个例子中，{}是一个占位符，用于表示待插入变量的位置。".format(epoch)"表示通过.format()方法将变量epoch的值插入到占位符的位置。所生成的最终字符串将包含"Epoch: "和epoch的值。
举个例子，如果epoch的值是10，那么该代码将生成字符串"Epoch: 10"。
'''
```

shuffle为false时两轮图片加载中随机选取结果相同

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day2202307201728587.png" alt="image-20230720172220235" style="zoom:50%;" />15 神经网络的基本骨架-nn.Module的使用

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212127626.png" alt="image-20230720204614822" style="zoom:67%;" />

## 自定义神经网络

### 重写方法

![image-20230720205054565](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212127438.png)

```python
import torch
from torch import nn

# 继承nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output
    
# 创建神经网络
tudui = Tudui()
# 创建输入
x = torch.tensor(1.0)
# 将x输入神经网络
output = tudui(x)
print(output)
```

# 16 卷积操作

卷积核移动，每个位置，卷积核的每一小块与输入图像重叠部分每一小块的相乘，所有乘积相加即为输出的一个小块

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128784.png" alt="image-20230720210532679" style="zoom:50%;" />

Stride为卷积核每次移动的步数

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128132.png" alt="image-20230720210651404" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128270.png" alt="image-20230720210746107" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128145.png" alt="image-20230720212346920" style="zoom:50%;" />

## 编写程序

```python
import  torch

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])
print(input.shape)
print(kernel.shape)
'''
一开始只有尺寸只有两个维度
torch.Size([5, 5])
torch.Size([3, 3])
'''
```

### 使用reshape()

```python
# batch_size, channel, 5*5
input = torch.reshape(input, (1,1,5,5))
kernel = torch.reshape(kernel, (1,1,3,3))
'''
torch.Size([1, 1, 5, 5])
torch.Size([1, 1, 3, 3])
'''
```

### 实现卷积操作

```python
import  torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])
# batch_size, channel, 5*5
input = torch.reshape(input, (1,1,5,5))
kernel = torch.reshape(kernel, (1,1,3,3))

print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
print(output)
'''
torch.Size([1, 1, 5, 5])
torch.Size([1, 1, 3, 3])
tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
'''
```

### 改变stride 步幅

```python
output = F.conv2d(input, kernel, stride=1)
print(output)
output2 = F.conv2d(input, kernel, stride=2)
print(output)
'''
tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
'''
```

## Padding 填充

图像周围填充0

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128205.png" alt="image-20230720212848323" style="zoom:67%;" />

```python
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
'''
tensor([[[[ 1,  3,  4, 10,  8],
          [ 5, 10, 12, 12,  6],
          [ 7, 18, 16, 16,  8],
          [11, 13,  9,  3,  4],
          [14, 13,  9,  7,  4]]]])
'''
```

# 17 神经网络-卷积层

## In_channel输入通道和Out_channel输出通道

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128005.png" alt="image-20230720221004485" style="zoom:50%;" />

### out_channel为2时

卷积操作完成后输出的 out_channels，取决于卷积核的数量。

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128788.png" alt="image-20230720221112952" style="zoom:50%;" />

## 编写代码验证

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        # super()函数通过传入当前类的名称（即Tudui）作为第一个参数，告诉Python去寻找并调用Tudui类的下一个父类的方法。
        super(Tudui, self).__init__()
        # 卷积层
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # x放入卷积层
        x = self.conv1(x)
        return x

tudui = Tudui()
print(tudui)
'''
Tudui(
  (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
)
'''
```

```python
# 查看每一个数据
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
 	print(imgs.shape)
    print(output.shape)
'''
batch_size为64.in_channel为3,out_channel为6
torch.Size([64, 3, 32, 32])
torch.Size([64, 6, 30, 30])
    '''
```

output为6个channel无法用writer显示，用reshape变为3个channel

```python
    # -1表示会自动计算
    torch.reshape(output, (-1 , 3, 30, 30))
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128048.png" alt="image-20230721105114448" style="zoom:50%;" />

# 18 最大池的使用

## ceil mode和floor mode

ceil mode是向上取整，floor mode是向下取整

具体到池化操作，ceil mode指如果池化核覆盖范围内有空缺，还是保留空缺继续池化；floor mode就会将空缺舍弃，不对其进行池化。

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128700.png" alt="image-20230721111639352" style="zoom:67%;" />

每次找出被池化核覆盖的范围内的最大值输出

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128281.png" alt="image-20230721111510866" style="zoom:50%;" />

步幅为kernel_size的大小，3

## Ceil mode

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212128639.png" alt="image-20230721111944689" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129643.png" alt="image-20230721112032005" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129583.png" alt="image-20230721112059463" style="zoom:50%;" />

默认情况ceil mode为false，即不保留

## 代码演示

```python
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],   # 设置为浮点数
                      [2,1,0,1,1]],dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output

tudui = Tudui()
output = tudui(input)
print(output)
'''
torch.Size([1, 1, 5, 5])
tensor([[[[2., 3.],
          [5., 1.]]]])
'''
```

### Ceil mode为False

```python
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)
'''
tensor([[[[2.]]]])
'''
```

## 最大池化的作用

保留数据特征，减小数据量

```python
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)



class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output

tudui = Tudui()
writer = SummaryWriter("logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129965.png" alt="image-20230721113804610" style="zoom:50%;" />

# 19 非线性激活

## 以ReLU为例



<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129815.png" alt="image-20230721114225891" style="zoom:50%;" />

### 参数inPlace

表示是否对原来变量进行变换，默认是False

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129845.png" alt="image-20230721114819121" style="zoom:67%;" />

```python
import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                     [-1, 3]])

# input需要指定batch_size
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu = ReLU()

    def forward(self, input):
        output = self.relu(input)
        return output

tudui = Tudui()
output = tudui(input)
print(output)
'''
torch.Size([1, 1, 2, 2])
tensor([[[[1., 0.],
          [0., 3.]]]])
'''
```

### Sigmoid函数

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129448.png" alt="image-20230721115922770" style="zoom:50%;" />

```python
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                     [-1, 3]])

# input需要指定batch_size
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

tudui = Tudui()

writer = SummaryWriter("logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129198.png" alt="image-20230721120003234" style="zoom:50%;" />

# 20 线性层及其它层介绍

## 线性层

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129928.png" alt="image-20230721142029222" style="zoom:50%;" />

5×5经过reshape变为1×25，再经过线性层变为1×3

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129920.png" alt="image-20230721142249064" style="zoom:67%;" />

```python
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 输入神经元个数，输出神经元个数
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1,1,1,-1))
    print(output.shape)
    output = tudui(output)
    print(output.shape)
    '''
 torch.Size([64, 3, 32, 32])
torch.Size([1, 1, 1, 196608])
torch.Size([1, 1, 1, 10])
# 输入batch_size为64，输出batch_size为1，就是说想用1张图片概括64张图片的特征？
    '''
```

### Flatten()函数

可以把输入展成一行,变为一维向量

![image-20230721143627486](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129710.png)

```python
    output = torch.flatten(imgs)
'''
torch.Size([64, 3, 32, 32])
torch.Size([196608])
torch.Size([10])
'''
```

# 21 搭建小实战和Sequential的使用

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129225.png" alt="image-20230721160609054" style="zoom:67%;" />

卷积的padding和stride可以用公式计算

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129852.png" alt="image-20230721154938067" style="zoom:67%;" />

padding为2，stride为1

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212129734.png" alt="image-20230721160010309" style="zoom:67%;" />

## 创建网络

```python
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, 2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        # 两个线性层
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

tudui = Tudui()
print(tudui)
'''
Tudui(
  (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
  (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear1): Linear(in_features=1024, out_features=64, bias=True)
  (linear2): Linear(in_features=64, out_features=10, bias=True)
)
'''
```

## 检查网络正确性

```python
# 元素都是1
input = torch.ones(64, 3, 32, 32)
output = tudui(input)
print(output.shape)
'''
# 每张图片对应10,64张图片
torch.Size([64, 10])
'''
```

## Sequential使用

代码更简洁

```python
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
```

## add_graph()显示训练过程

```python
tudui = Tudui()
print(tudui)

# 元素都是1
input = torch.ones(64, 3, 32, 32)
output = tudui(input)
print(output.shape)

writer = SummaryWriter("logs2")
writer.add_graph(tudui, input)
writer.close()
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130816.png" alt="image-20230721162855578" style="zoom:67%;" />

双击查看细节

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130215.png" alt="image-20230721162914891" style="zoom:50%;" />

# 22 损失函数与反向传播

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130826.png" alt="image-20230721164558100" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130961.png" alt="image-20230721164717441" style="zoom:67%;" />

## L1loss 函数

![image-20230721164852676](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130158.png)

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130420.png" alt="image-20230721164934328" style="zoom:50%;" />

```python
import torch
from torch import float32
from torch.nn import L1Loss

inputs = torch.tensor([1,2,3], dtype=torch.float32)
targets = torch.tensor([1,2,5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1,1,1,3))
targets = torch.reshape(targets, (1,1,1,3))

loss = L1Loss()
result = loss(inputs, targets)

print(result)
'''
tensor(0.6667)
'''
```

### 改变reduction

```python
# 结果为差距总和
loss = L1Loss(reduction='sum')
'''
tensor(2.)
'''
```

## MSELOSS 平方差

```python
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)
# tensor(1.3333)
```



![image-20230721165853991](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130621.png)

## 交叉熵

分类问题。下图的log应该是ln

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130704.png" alt="image-20230721171349447" style="zoom:50%;" />

```python
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1,3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
# tensor(1.1019)
```

## 查看输出和target

```python
import torch
import torchvision
from torch import float32, nn
from torch.nn import L1Loss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


tudui = Tudui()
for data in dataloader:
        imgs, targets = data
        output = tudui(imgs)
        print(output) 
        print(targets)
'''
前者为每个类别的概率，后者为target：3
tensor([[ 0.0277,  0.1381, -0.0236,  0.0042, -0.1030, -0.0837, -0.0184,  0.0114,
          0.1186,  0.0049]], grad_fn=<AddmmBackward0>)
tensor([3])
'''
```

### 添加交叉熵

```python
loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
        imgs, targets = data
        output = tudui(imgs)
        result_loss = loss(output, targets)
        print(result_loss)
 # tensor(2.4989, grad_fn=<NllLossBackward0>)
```

## 梯度下降法

```python
loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
        imgs, targets = data
        output = tudui(imgs)
        result_loss = loss(output, targets)
        result_loss.backward()
        print("ok")
```

# 23 优化器（一）

```python
import torch
import torchvision
from torch import float32, nn
from torch.nn import L1Loss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
# 随机梯度下降
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
            imgs, targets = data
            output = tudui(imgs)
            result_loss = loss(output, targets)
            # 梯度清零
            optim.zero_grad()
            # 反向传播
            result_loss.backward()
            # 优化
            optim.step()
            # 每一轮所有数据损失总和
            running_loss = running_loss + result_loss
    print(running_loss)
'''
tensor(18669.0332, grad_fn=<AddBackward0>)
tensor(16020.8330, grad_fn=<AddBackward0>)
'''
```

# 24 现有网络模型的使用及修改

## VGG16

最后out_feature为1000，表明1000个分类

```python
import torchvision
from torch import nn

# 加载网络模型，不用下载
vgg16_false = torchvision.models.vgg16(pretrained=False)
# 下载训练好的参数
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)
'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''
```

给vgg16多添加一个线性层，实现10个分类

```python
vgg16_true.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)
'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
  (add_linear): Linear(in_features=1000, out_features=10, bias=True)
)
'''
```

将线性层加到classifier中

```python
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)
'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
    (add_linear): Linear(in_features=1000, out_features=10, bias=True)
  )
)
'''
```

```python
# 修改原有的最后一层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print((vgg16_false))
'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
)
'''
```

# 25 网络模型的保存与读取

## 保存vgg16

```python
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式一
torch.save(vgg16, "vgg16_method1.pth")
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307212130455.png" alt="image-20230721210321691" style="zoom:67%;" />

## 加载模型

```python
import torch

# 保存方式1(保存模型结构和参数)，加载模型

model = torch.load("vgg16_method1.pth")
print(model)
```

## 保存方式2

```python
# 保存方式2
# 不保存结构，保存参数，保存为字典，推荐使用
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
model2 = torch.load("vgg16_method2.pth")
print(model2)
'''
OrderedDict([('features.0.weight', tensor([[[[-0.0040,  0.0666, -0.1964],
          [ 0.0534, -0.0111, -0.0529],
          [-0.0224, -0.1023, -0.1115]],

         [[ 0.0481,  0.0253, -0.0616],
          [-0.0166, -0.0122, -0.0387],
          [ 0.0031, -0.0336,  0.0157]],

         [[ 0.0209,  0.0349,  0.0231],
          [-0.0072, -0.0687, -0.0050],
          [-0.0395,  0.0666,  0.1481]]],
'''
```

## 恢复成网络模型

新建网路模型结构

```python
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''
```

## 方式1陷阱

保存模型

```python
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")
```

加载时报错

```python
model = torch.load("tudui_method1.pth")
print(model)
# AttributeError: Can't get attribute 'Tudui' on <module '__main__' from 'D:\\PytorchLearning\\model_save.py'>
```

需要将模型的定义放在需要加载的文件

```python
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x
model = torch.load("tudui_method1.pth")
print(model)
```

# 26 完整的模型训练

## Argmax

输入两张图片，通过outputs得到预测类别Preds

将Preds与Inputs target比较。

[false, true].sum()=1，false看成0，true看成1

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222023245.png" alt="image-20230722155229401" style="zoom:67%;" />

```python
import torch

outputs = torch.tensor([[0.1,0.2],
                       [0.3,0.4]])
# 1表示横向看
print(outputs.argmax(1))
# tensor([1, 1])
```

![image-20230722155601779](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024624.png)

![image-20230722155647986](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024441.png)

```python
import torch

outputs = torch.tensor([[0.1,0.2],
                       [0.3,0.4]])
# 1表示横向看
preds = outputs.argmax(1)
targets = torch.tensor([0, 1])
print(preds == targets)
print((preds == targets).sum())
'''
tensor([False,  True])
tensor(1)
'''
```

## 完整代码

### model.py

```python
# 搭建神经网络
import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)

```

### 训练和测试代码

```python
import  torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root = "./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root = "./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 字符串格式化
# 如果train_size = 10, 训练数据集长度为：10
print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 创建网络模型
tudui = Tudui()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 科学计数法
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")
for i in range(epoch):
    print("-----------第 {} 轮训练开始----------".format(i + 1))
    # 训练开始
    for data in train_dataloader:
        imgs, targets = data
        output = tudui(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # item()可以把形如tensor(5)的类型转换成数字5
        # 每100步骤打印
        if total_train_step % 100 == 0:
            print("训练次数 {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 不需要调优，取消梯度
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    # 总正确率/测试集长度
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step = total_test_step + 1
    # 保存每一个epoch的结果
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
'''
-----------第 10 轮训练开始----------
训练次数 7100, Loss: 1.28587007522583
训练次数 7200, Loss: 0.9727596640586853
训练次数 7300, Loss: 1.1144425868988037
训练次数 7400, Loss: 0.8997210264205933
训练次数 7500, Loss: 1.2414882183074951
训练次数 7600, Loss: 1.2502361536026
训练次数 7700, Loss: 0.8779300451278687
训练次数 7800, Loss: 1.2825963497161865
整体测试集上的Loss: 194.9348732829094
整体测试集上的正确率: 0.5584999918937683
模型已保存
'''
```

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024261.png" alt="image-20230722162840740" style="zoom:67%;" />

## 注意点

### train()和eval()

```python
# 训练开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        output = tudui(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # item()可以把形如tensor(5)的类型转换成数字5
        # 每100步骤打印
        if total_train_step % 100 == 0:
            print("训练次数 {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 不需要调优，取消梯度
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    # 总正确率/测试集长度
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step = total_test_step + 1
    # 保存每一个epoch的结果
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")
```

# 27 利用GPU训练(一)

![image-20230722163612035](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024674.png)

```python
import  torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import  time

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root = "./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root = "./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 字符串格式化
# 如果train_size = 10, 训练数据集长度为：10
print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 创建网络模型
# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)


tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
# learning_rate = 0.01
# 科学计数法
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("-----------第 {} 轮训练开始----------".format(i + 1))
    # 训练开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = tudui(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # item()可以把形如tensor(5)的类型转换成数字5
        # 每100步骤打印
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数 {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 不需要调优，取消梯度
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
               imgs = imgs.cuda()
               targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    # 总正确率/测试集长度
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step = total_test_step + 1
    # 保存每一个epoch的结果
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
'''
-----------第 1 轮训练开始----------
6.235116958618164
训练次数 100, Loss: 2.292055368423462
'''
```

## Goole Colaboratory 

打开GPU

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024730.png" alt="image-20230722165519327" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024908.png" alt="image-20230722165546244" style="zoom:50%;" />

代码前加！表示不用python语法，用终端语法

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024633.png" alt="image-20230722165905600" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024827.png" alt="image-20230722165935223" style="zoom:50%;" />

# 28 利用GPU训练(二)

<img src="https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222024381.png" alt="image-20230722170622579" style="zoom:67%;" />

```python
import  torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time

# 定义训练的设备
device = torch.device("cpu")
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root = "./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root = "./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 字符串格式化
# 如果train_size = 10, 训练数据集长度为：10
print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 创建网络模型
# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)


tudui = Tudui()
tudui = tudui.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
# learning_rate = 0.01
# 科学计数法
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("-----------第 {} 轮训练开始----------".format(i + 1))
    # 训练开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = tudui(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # item()可以把形如tensor(5)的类型转换成数字5
        # 每100步骤打印
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数 {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 不需要调优，取消梯度
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    # 总正确率/测试集长度
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step = total_test_step + 1
    # 保存每一个epoch的结果
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
```

### device写法

```python
device = torch.device("cuda")
device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

# 29 完整模型验证

```python
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./img/dog.png"
image = Image.open(image_path)
print(image)
# png是四个通道，除了RGB三通道外还有一个透明度通道
# 调用convert保留其颜色通道
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
# 加载模型
# cpu上想使用gpu训练的模型需要映射
model = torch.load("tudui_0.pth", map_location=torch.device('cpu'))
print(model)

image = torch.reshape(image, (1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
```

# 30 阅读开源项目

[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

```python
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

```

![image-20230722201917644](https://cdn.jsdelivr.net/gh/techniquenotes/photohouse@main/Pytorch/Learning%20note/Day3/202307222025813.png)

requered=True表明一定需要这个参数

可以把其改成default，就可以在pycharm中右键运行

```python
# 修改后
     parser.add_argument('--dataroot', default="./dataset/maps", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
```

