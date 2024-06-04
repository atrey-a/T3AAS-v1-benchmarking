import os
import torch
from torch.utils.data import Dataset
import pandas
import numpy as np
from scipy.stats import skew as Skewness, kurtosis as Kurtosis
from torchvision import transforms
from PIL import Image


class SVC(Dataset):
    def __init__(self, dataset_path, loader_device):
        super().__init__()
        self.loader_device = loader_device
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in sorted(os.listdir(os.path.join(dataset_path,foldername))):
                self.filelist.append(os.path.join(dataset_path,foldername,filename))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        filepath = self.filelist[idx]
        foldername = os.path.basename(os.path.dirname(filepath))
        label = self.folderlist.index(foldername)

        tensor = torch.from_numpy(
            pandas.read_csv(filepath, header=None,sep=",").to_numpy()
        ).to(self.loader_device).type(torch.float32)

        return tensor, label


class SVC_Features(Dataset):
    def __init__(
        self,
        dataset_path,
        loader_device,
        num_columns,
        num_windows,
        window_size,
        minimum=True,
        maximum=True,
        mean=True,
        median=True,
        skewness=True,
        kurtosis=True,
        std=False
    ):
        super().__init__()
        self.loader_device = loader_device
        self.num_columns = num_columns
        self.num_windows = num_windows
        self.window_size = window_size
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in os.listdir(os.path.join(dataset_path,foldername)):
                    self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.median = median
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.std = std
        self.num_descriptors = sum(int(desc) for desc in [self.minimum,self.maximum,self.mean,self.median,self.skewness,self.kurtosis,self.std])

        self.list_of_all = []
        self.full_length = len(self.filelist)

    def descriptors(self, tensor_list):
        descriptors_tensor = torch.zeros([self.num_windows,self.num_descriptors,self.num_columns]).to(self.loader_device)
        v = torch.zeros([1])
        for i,tensor in enumerate(tensor_list):
            descriptors_ = torch.zeros([self.num_descriptors])
            for k in range(self.num_columns):
                j = 0
                if self.maximum:
                    descriptors_[j],_ = torch.max(tensor[:,k],0)
                    j += 1
                if self.mean:
                    descriptors_[j] = torch.mean(tensor[:,k],dim=0)
                    j += 1
                if self.median:
                    descriptors_[j],_ = torch.median(tensor[:,k],dim=0)
                    j += 1
                if self.skewness:
                    v[0] = Skewness(tensor[:,k].cpu(),nan_policy='raise')
                    if not torch.isnan(v[0]): descriptors_[j] = v[0]
                    j += 1
                if self.kurtosis:
                    v[0] = Kurtosis(tensor[:,k].cpu(),nan_policy='raise')
                    if not torch.isnan(v[0]): descriptors_[j] = v[0]
                    j += 1
                descriptors_tensor[i,:,k] = descriptors_
        return descriptors_tensor

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, idx):
        filepath = self.filelist[idx]
        foldername = os.path.basename(os.path.dirname(filepath))
        label = self.folderlist.index(foldername)

        tensor = torch.from_numpy(
            np.genfromtxt(filepath,delimiter=',')
        ).to(self.loader_device).type(torch.float32)

        start_seq = [
            round((i * (tensor.shape[0] - self.window_size)) / (self.num_windows - 1))
            for i in range(self.num_windows)
        ]
        out = [
            tensor[start_seq[idx] : start_seq[idx] + self.window_size, :]
            for idx in range(self.num_windows)
        ]
        out = self.descriptors(out)

        return out, label


class SVC_Images(Dataset):
    def __init__(
        self,
        dataset_path,
        loader_device,
    ):
        super().__init__()
        self.loader_device = loader_device
        self.folderlist = sorted(list(os.listdir(dataset_path)))
        self.filelist = []
        for foldername in self.folderlist:
            for filename in os.listdir(os.path.join(dataset_path,foldername)):
                    self.filelist.append(os.path.join(dataset_path,foldername,filename))
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.to(self.loader_device)
                ),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1)
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.list_of_all)

    def __getitem__(self, idx):
        filepath = self.filelist[idx]
        foldername = os.path.basename(os.path.dirname(filepath))
        label = self.folderlist.index(foldername)

        x = Image.open(filepath).convert('L')
        x = self.transform_image(x).type(torch.float32)

        return x, label
