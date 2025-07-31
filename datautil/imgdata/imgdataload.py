import numpy as np
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_dataset(image_list, labels, append_root=None):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            if append_root:
                images = [((append_root + val.split()[0]), int(val.split()[1])) for val in image_list]
            else:
                images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


class ImageDataset(object):
    def __init__(self, dataset, root_dir, domain_name, labels= None, transform=None, 
                    target_transform= None, indices= None, mode= 'Default',
                    return_idx=False, transform2=None,) -> None:
        if not dataset in ['domainnet126','office']:
            self.imgs, self.class_to_idx = ImageFolder(root_dir + domain_name).imgs, ImageFolder(root_dir + domain_name).class_to_idx
        else:
            txt = open(root_dir+domain_name+'_list.txt').readlines()
            self.imgs = make_dataset(txt, labels, append_root=root_dir)
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        if labels is None:
            labels = [item[1] for item in self.imgs]
        self.labels = np.array(labels)
        self.x = imgs
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.return_idx = return_idx
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        
    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y
    
    def input_trans(self, x):
        if self.transform is not None and self.transform2 is not None:
            return self.transform(x), self.transform2(x)
        if self.transform is not None:
            return self.transform(x)
        else: 
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img_q = self.input_trans(self.loader(self.x[index]))   # the quere image in CO
        ctarget = self.target_trans(self.labels[index])

        if self.return_idx:
            return img_q, ctarget, index
        return img_q, ctarget

    def __len__(self):
        return len(self.indices)
    
