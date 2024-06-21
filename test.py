import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
# 定义数据集类
class MyData(Dataset):

    def __init__(self, root_dir, laber_dir):
        self.root_dir = root_dir
        self.label_dir = laber_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)

# 数据集根目录
root_dir="hymenoptera_data"

# 各个数据集的创建
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

# 数据集整合
train_dataset = ants_dataset+bees_dataset


#初始化tensorboard记录器
writer = SummaryWriter("logs")
image_path = "data/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)


#数据转化
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img_PIL)


# 调用transforms反馈训练图像
writer.add_image("Tensor_img",tensor_img)

writer.close()

# 调用tensorboard描绘训练结果的反馈图像
writer.add_image("test", img_array,2, dataformats='HWC')
# y=x
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()



# 训练循环，num_epochs为训练次数

num_epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_freq = 10

# train_loader = os.listdir(path)
# 定义模型、优化器和损失函数(尚未确定）
model = ...
optimizer = ...
criterion = ...

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_dataset.datasets):
        # 假设images和labels已经准备好，将它们送入设备
        images = images.to(device)
        labels = labels.to