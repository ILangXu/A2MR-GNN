import torch
import os
import pickle
from torch.utils.data import Dataset
from dataloader import DataLoader
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import (RandomFlip,
                                        RandomRotate,
                                        RandomTranslate,
                                        Compose,
                                        Center)

path = 'data/mice_features/test/all_5A/'
labels = 'data/mice_features/total_labels.pkl'

class GINDataset(Dataset):

    def __init__(self, root=path, label=labels, phase='train'):
        'Initialization'
        # root = os.path.join(root, phase)
        total = os.listdir(root) #所有数据的列表
        self.root = root
        self.data = total #所有数据列表
        self.phase = phase
        np.random.seed(16)
        with open(label, 'rb') as f:
            self.labels = pickle.load(f) #读取标签信息

        total_size = len(total)
        permu = np.random.permutation(total_size)#把输入数据随机排列
        if phase == 'train':
            self.list_IDs = permu[:int(total_size*0.9)]
        elif phase == 'val':
            self.list_IDs = permu[int(total_size*0.9):]
        elif phase == 'test':
            self.list_IDs = permu[:]
        else:
            raise ValueError('wrong phase!')
        self.transform = Compose([RandomFlip(0), #数据增强的方法
                                  RandomFlip(1),
                                  RandomFlip(2),
                                  RandomRotate(360, 0),
                                  RandomRotate(360, 1),
                                  RandomRotate(360, 2),
                                  Center()])
        self.trans = Center()

    def __len__(self):
        'Denotes the total number of samples'
        return self.list_IDs.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        path = self.data[index] #根据索引提取对应的文件名
        # label = convert_y_unit(label, 'nM', 'p')
        path = os.path.join(self.root, path) #完整的文件名路径：data/mice_features/all/1a4k.pkl
        with open(path, 'rb') as f:
            x, edge_index_inner, edge_index_out, edge_attr_inner,edge_attr_outer = pickle.load(f)#获取特征信息
        label = self.labels[path.split('/')[-1].split('.')[0]]#根据key获取value标签值，先用/分割字符串获取最后一个字符也就是文件名1a4k.pkl，再用.分割，得到第一个字符，就是复合体的名称1a4k
        edge_index = torch.cat((edge_index_inner, edge_index_out), dim=1)
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_index_inner=edge_index_inner,
                    edge_index_out=edge_index_out,
                    edge_attr_inner = edge_attr_inner,
                    edge_attr_outer=edge_attr_outer,
                    pos=x[:, -3:],
                    )
        if type(label) not in [int, float]:
            print(label)
        if self.phase == 'train':
            self.transform(data)
        else:
            self.trans(data)
        return data, torch.tensor(label).float()

def get_gin_dataloader(path, label_path, batch_size,
                       num_workers=6, phase='train'):
    dataset = GINDataset(path, label_path, phase=phase)#返回的是节点特征矩阵，边索引，边特征矩阵，标签信息
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle)
    return dataloader
