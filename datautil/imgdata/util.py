from torchvision import transforms
from PIL import Image, ImageFile, ImageFilter
import random
import torch
def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def image_train(dg_aug=False, resize_size=256, crop_size=224 ):
    normalize = transforms.Normalize(mean= [0.485, 0.456, 0.406], 
                                    std= [0.229, 0.224, 0.225])
    if not dg_aug:
        train_transform= [
            transforms.Resize((resize_size, resize_size)), 
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            normalize
        ]
    else:
        train_transform = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), 
            transforms.RandomGrayscale(), 
            transforms.ToTensor(), 
            normalize
        ]
    return transforms.Compose(train_transform)


def img_test(dg_aug=False, resize= 256, crop_size= 224):
    normalize = transforms.Normalize(mean= [0.485, 0.456, 0.406], 
                                    std= [0.229, 0.224, 0.225])
    if not dg_aug:
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop_size), 
            transforms.ToTensor(), 
            normalize
        ])    
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        normalize
    ])

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
