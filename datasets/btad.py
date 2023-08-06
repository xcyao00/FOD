import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


BTAD_CLASS_NAMES = ['01', '02', '03']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BTADDataset(Dataset):
    def __init__(self, 
                 data_path,
                 classname,
                 resize=256,
                 cropsize=256,
                 is_train=True):
        assert classname in BTAD_CLASS_NAMES, 'class_name: {}, should be in {}'.format(classname, BTAD_CLASS_NAMES)
        self.dataset_path = data_path
        self.class_name = classname
        self.is_train = is_train
        self.cropsize = cropsize
        # load dataset
        self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(resize, Image.ANTIALIAS),
                #T.RandomRotation(5),
                T.CenterCrop(cropsize),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(resize, Image.ANTIALIAS),
                T.CenterCrop(cropsize),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(resize, Image.NEAREST),
            T.CenterCrop(cropsize),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __getitem__(self, idx):
        img_path, y, mask, img_type = self.x[idx], self.y[idx], self.mask[idx], self.img_types[idx]

        x = Image.open(img_path).convert('RGB')
        
        x = self.normalize(self.transform_x(x))
        
        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, y, mask, os.path.basename(img_path[:-4]), img_type
    
    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, types = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.bmp') or f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'ok':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                types.extend(['ok'] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                if self.class_name == '03':
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.bmp')
                                 for img_fname in img_fname_list]
                else:
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                    for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask), list(types)