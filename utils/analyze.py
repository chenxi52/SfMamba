import os
import logging
import time
import math
from functools import partial
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from collections import OrderedDict
import cv2
from matplotlib.colors import Normalize

import tqdm
from PIL import Image
import os

import torch
import torch.nn as nn
from functools import partial
from typing import Callable, Tuple, Union, Tuple, Union, Any

HOME = os.environ["HOME"].rstrip("/")

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


def get_dataset(root="./val", img_size=224, ret="", crop=True, single_image=False):
    from torch.utils.data import SequentialSampler, DistributedSampler, DataLoader
    size = int((256 / 224) * img_size) if crop else int(img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    ])
    if single_image:
        class ds(datasets.ImageFolder):
            def __init__(self, img, transform):
                self.transform = transform
                self.target_transform = None
                self.loader = datasets.folder.default_loader
                self.samples = [(img, 0)]
                self.targets = [0]
                self.classes = ["none"]
                self.class_to_idx = {"none": 0}
        dataset = ds(root, transform=transform)
    else:
        dataset = datasets.ImageFolder(root, transform=transform)
        if ret in dataset.classes:
            print(f"found target {ret}", flush=True)
            target = dataset.class_to_idx[ret]
            dataset.samples =  [s for s in dataset.samples if s[1] == target]
            dataset.targets = [s for s in dataset.targets if s == target]
            dataset.classes = [ret]
            dataset.class_to_idx = {ret: target}
    return dataset


def show_mask_on_image(img: torch.Tensor, mask: torch.Tensor, mask_norm=True):
    H, W, C = img.shape
    mH, mW = mask.shape
    mask = torch.nn.functional.interpolate(mask[None, None], (H, W), mode="bilinear")[0, 0]
    if mask_norm:
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    img = img.clamp(min=0, max=1).cpu().numpy()
    mask = mask.clamp(min=0, max=1).cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    return heatmap
    return np.uint8(255 * cam)


def get_val_dataloader(batch_size=64, root="./val", img_size=224, sequential=True):
    import torch.utils.data
    size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset = datasets.ImageFolder(root, transform=transform)
    if sequential:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.DistributedSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


class visualize:
    @staticmethod
    def get_colormap(name):
        import matplotlib as mpl
        """Handle changes to matplotlib colormap interface in 3.6."""
        try:
            return mpl.colormaps[name]
        except AttributeError:
            return mpl.cm.get_cmap(name)

    @staticmethod
    def draw_image_grid(image: Image, grid=[(0, 0, 1, 1)], **kwargs):
        # grid[0]: (x,y,w,h)
        default = dict(fill=None, outline='red', width=3)
        default.update(kwargs)
        assert isinstance(grid, list) and isinstance(grid[0], tuple) and len(grid[0]) == 4
        from PIL import ImageDraw
        a = ImageDraw.ImageDraw(image)
        for g in grid:
            a.rectangle([(g[0], g[1]), (g[0] + g[2], g[1] + g[3])], **default)
        return image

    @staticmethod
    def visualize_attnmap(attnmap, savefig="", figsize=(18, 16), cmap=None, sticks=True, dpi=400, 
                          fontsize=35, colorbar=True, image=None, **kwargs):
        import matplotlib.pyplot as plt
        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()
        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        if image is not None:
            image = image.cpu().numpy()
        # attnmap = np.resize(attnmap, image.shape)
        attnmap_resized = cv2.resize(attnmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        norm = Normalize(vmin=attnmap_resized.min(), vmax=attnmap_resized.max())
        normalized_attn = norm(attnmap_resized)

        cmap = plt.get_cmap('jet')  # 或其他颜色映射
        colored_attn = cmap(normalized_attn)

        # 混合原图和注意力图
        blended = (1 - 0.3) * image + 0.3 * colored_attn[..., :3]  # 30%注意力图混合
        ax.imshow(blended)
        ax.set_axis_off()
        plt.savefig(savefig)
        plt.close()


    @staticmethod
    def visualize_attnmaps(attnmaps, savefig="", figsize=(18, 16), rows=1, cmap=None, dpi=400, 
                           fontsize=35, linewidth=2, **kwargs):
        # attnmaps: [(map, title), (map, title),...]
        import math
        import matplotlib.pyplot as plt
        vmin = min([np.min((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        vmax = max([np.max((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        cols = math.ceil(len(attnmaps) / rows)
        plt.rcParams["font.size"] = fontsize
        figsize=(cols * figsize[0], rows * figsize[1])
        fig, axs = plt.subplots(rows, cols, squeeze=False, sharex="all", sharey="all", figsize=figsize, dpi=dpi)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    image = np.zeros_like(image)
                    title = "pad"
                else:
                    image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                im = axs[i, j].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
                axs[i, j].set_title(title)
                axs[i, j].set_yticks([])
                axs[i, j].set_xticks([])
                print(title, "max", np.max(image), "min", np.min(image), end=" | ")
            print("")
        axs[0, 0].figure.colorbar(im, ax=axs)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()
        print("")

    @staticmethod
    def seanborn_heatmap(
        data, *,
        vmin=None, vmax=None, cmap=None, center=None, robust=False,
        annot=None, fmt=".2g", annot_kws=None,
        linewidths=0, linecolor="white",
        cbar=True, cbar_kws=None, cbar_ax=None,
        square=False, xticklabels="auto", yticklabels="auto",
        mask=None, ax=None,
        **kwargs
    ):
        from matplotlib import pyplot as plt
        from seaborn.matrix import _HeatMapper
        # Initialize the plotter object
        plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                            annot_kws, cbar, cbar_kws, xticklabels,
                            yticklabels, mask)

        # Add the pcolormesh kwargs here
        kwargs["linewidths"] = linewidths
        kwargs["edgecolor"] = linecolor

        # Draw the plot and return the Axes
        if ax is None:
            ax = plt.gca()
        if square:
            ax.set_aspect("equal")
        plotter.plot(ax, cbar_ax, kwargs)
        mesh = ax.pcolormesh(plotter.plot_data, cmap=plotter.cmap, **kwargs)
        return ax, mesh


    @classmethod
    def visualize_snsmap(cls, attnmap, savefig="", figsize=(18, 16), cmap=None, sticks=True, dpi=80, fontsize=35, linewidth=2, **kwargs):
        import matplotlib.pyplot as plt
        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()
        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        _, mesh = cls.seanborn_heatmap(attnmap, xticklabels=sticks, yticklabels=sticks, cmap=cmap, linewidths=0,
                center=0, annot=False, ax=ax, cbar=False, annot_kws={"size": 24}, fmt='.2f')
        cb = ax.figure.colorbar(mesh, ax=ax)
        cb.outline.set_linewidth(0)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()


    @classmethod
    def visualize_snsmaps(cls, attnmaps, savefig="", figsize=(18, 16), rows=1, cmap=None, sticks=True, dpi=80, fontsize=35, linewidth=2, **kwargs):
        # attnmaps: [(map, title), (map, title),...]
        import math
        import matplotlib.pyplot as plt
        vmin = min([np.min((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        vmax = max([np.max((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        cols = math.ceil(len(attnmaps) / rows)
        plt.rcParams["font.size"] = fontsize
        figsize=(cols * figsize[0], rows * figsize[1])
        fig, axs = plt.subplots(rows, cols, squeeze=False, sharex="all", sharey="all", figsize=figsize, dpi=dpi)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    image = np.zeros_like(image)
                    title = "pad"
                else:
                    image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                _, im = cls.seanborn_heatmap(image, xticklabels=sticks, yticklabels=sticks, 
                                             vmin=vmin, vmax=vmax, cmap=cmap,
                                             center=0, annot=False, ax=axs[i, j], 
                                             cbar=False, annot_kws={"size": 24}, fmt='.2f')
                axs[i, j].set_title(title)
        cb = axs[0, 0].figure.colorbar(im, ax=axs)
        cb.outline.set_linewidth(0)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()


#  used for visualizing effective receiptive field 
class EffectiveReceiptiveField:
    @staticmethod
    def simpnorm(data):
        data = np.power(data, 0.2)
        data = data / np.max(data)
        return data

    @staticmethod
    def get_rectangle(data, thresh):
        h, w = data.shape
        all_sum = np.sum(data)
        for i in range(1, h // 2):
            selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
            area_sum = np.sum(selected_area)
            if area_sum / all_sum > thresh:
                return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
        return None, None

    @staticmethod
    def get_input_grad(model, samples, square=True):
        outputs = model(samples)
        out_size = outputs.size()
        if square:
            assert out_size[2] == out_size[3], out_size
        central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    @classmethod
    def get_input_grad_avg(cls, model: nn.Module, size=1024, data_path="ImageNet", num_images=50, norms=lambda x:x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        import tqdm
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, RandomSampler
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform)
        data_loader_val = DataLoader(dataset, sampler=RandomSampler(dataset), pin_memory=True)

        meter = AverageMeter()
        model.cuda().eval()
        for _, (samples, _) in tqdm.tqdm(enumerate(data_loader_val)):
            if meter.count == num_images:
                break
            samples = samples.cuda(non_blocking=True).requires_grad_()
            contribution_scores = cls.get_input_grad(model, samples)
            if np.isnan(np.sum(contribution_scores)):
                print("got nan | ", end="")
                continue
            else:
                meter.update(contribution_scores)
        return norms(meter.avg)


# used for visualizing the attention of mamba
class AttnMamba:
    @staticmethod
    def convert_state_dict_from_mmdet(state_dict):
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.startswith("backbone."):
                new_state_dict[k[len("backbone."):]] = state_dict[k]
        return new_state_dict

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

    @staticmethod
    @torch.no_grad()
    def attnmap_mamba(regs, mode="CB", ret="all", absnorm=0, scale=1, verbose=False, device=None):
        printlog = print if verbose else lambda *args, **kwargs: None
        print(f"attn for mode={mode}, ret={ret}, absnorm={absnorm}, scale={scale}", flush=True)

        _norm = lambda x: x
        if absnorm == 1:
            _norm = lambda x: ((x - x.min()) / (x.max() - x.min()))
        elif absnorm == 2:
            _norm = lambda x: (x.abs() / x.abs().max())

        As, Bs, Cs, Ds = -torch.exp(regs["A_logs"].to(torch.float32)), regs["Bs"], regs["Cs"], regs["Ds"]
        us, dts, delta_bias = regs["us"], regs["dts"], regs["delta_bias"]
        ys, oy = regs["ys"], regs["y"]
        H, W = regs["H"], regs["W"]
        printlog(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)
        B, G, N, L = Bs.shape
        GD, N = As.shape
        D = GD // G
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        if device is not None:
            As, Bs, Cs, Ds, us, dts, delta_bias, ys, oy = As.to(device), Bs.to(device), Cs.to(device), Ds.to(device), us.to(device), dts.to(device), delta_bias.to(device), ys.to(device), oy.to(device)

        mask = torch.tril(dts.new_ones((L, L)))
        dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, G, D, L)
        dw_logs = As.view(G, D, N)[None, :, :, :, None] * dts[:,:,:,None,:] # (B, G, D, N, L)
        ws = torch.cumsum(dw_logs, dim=-1).exp()

        if mode == "CB":
            Qs, Ks = Cs[:,:,None,:,:], Bs[:,:,None,:,:]
        elif mode == "CBdt":
            Qs, Ks = Cs[:,:,None,:,:], Bs[:,:,None,:,:] * dts.view(B, G, D, 1, L)
        elif mode == "CwBw":
            Qs, Ks = Cs[:,:,None,:,:] * ws, Bs[:,:,None,:,:] / ws.clamp(min=1e-20)
        elif mode == "CwBdtw":
            Qs, Ks = Cs[:,:,None,:,:] * ws, Bs[:,:,None,:,:]  * dts.view(B, G, D, 1, L) / ws.clamp(min=1e-20)
        elif mode == "ww":
            Qs, Ks = ws, 1 / ws.clamp(min=1e-20)
        else:
            raise NotImplementedError

        printlog(ws.shape, Qs.shape, Ks.shape)
        printlog("Bs", Bs.max(), Bs.min(), Bs.abs().min())
        printlog("Cs", Cs.max(), Cs.min(), Cs.abs().min())
        printlog("ws", ws.max(), ws.min(), ws.abs().min())
        printlog("Qs", Qs.max(), Qs.min(), Qs.abs().min())
        printlog("Ks", Ks.max(), Ks.min(), Ks.abs().min())
        _Qs, _Ks = Qs.view(-1, N, L), Ks.view(-1, N, L)
        attns = (_Qs.transpose(1, 2) @ _Ks).view(B, G, -1, L, L)
        attns = attns.mean(dim=2) * mask

        attn0 = attns[:, 0, :].view(B, -1, L, L)
        attn1 = attns[:, 1, :].view(-1, H, W, H, W).permute(0, 2, 1, 4, 3).contiguous().view(B, -1, L, L)
        attn2 = attns[:, 2, :].view(-1, L, L).flip(dims=[-2]).flip(dims=[-1]).contiguous().view(B, -1, L, L)
        attn3 = attns[:, 3, :].view(-1, L, L).flip(dims=[-2]).flip(dims=[-1]).contiguous().view(B, -1, L, L)
        attn3 = attn3.view(-1, H, W, H, W).permute(0, 2, 1, 4, 3).contiguous().view(B, -1, L, L)

        # ao0, ao1, ao2, ao3: attntion in four directions without rearrange
        # a0, a1, a2, a3: attntion in four directions with rearrange
        # a0a2, a1a3, a0a1: combination of "a0, a1, a2, a3"
        # all: combination of all "a0, a1, a2, a3"
        if ret in ["ao0"]:
            attn = _norm(attns[:, 0, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao1"]:
            attn = _norm(attns[:, 1, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao2"]:
            attn = _norm(attns[:, 2, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao3"]:
            attn = _norm(attns[:, 3, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["a0"]:
            attn = _norm(attn0).mean(dim=1)
        elif ret in ["a1"]:
            attn = _norm(attn1).mean(dim=1)
        elif ret in ["a2"]:
            attn = _norm(attn2).mean(dim=1)
        elif ret in ["a3"]:
            attn = _norm(attn3).mean(dim=1)
        elif ret in ["all"]:
            attn = _norm((attn0 + attn1 + attn2 + attn3)).mean(dim=1)
        elif ret in ["nall"]:
            attn = (_norm(attn0) + _norm(attn1) + _norm(attn2) + _norm(attn3)).mean(dim=1) / 4.0
        else:
            raise NotImplementedError(f"{ret} is not allowed")
        attn = (scale * attn).clamp(max=attn.max())
        return attn[0], H, W


    @classmethod
    @torch.no_grad()
    def get_attnmap_mamba(cls, ss2ds, stage=-1, mode="", verbose=False, raw_attn=False, block_id=0, scale=1, device=None):
        mode1 = mode.split("_")[-1]
        mode = mode[:-(len(mode1) + 1)]
        
        absnorm = 0
        tag, mode = cls.checkpostfix("_absnorm", mode)
        absnorm = 2 if tag else absnorm
        tag, mode = cls.checkpostfix("_norm", mode)
        absnorm = 1 if tag else absnorm

        if raw_attn:
            # ss2d = ss2ds if not isinstance(ss2ds, list) else ss2ds[stage][block_id]
            regs = getattr(ss2ds, "__data__")
            attn, H, W = cls.attnmap_mamba(regs, mode=mode1, ret=mode, absnorm=absnorm, verbose=verbose, scale=scale)
            return attn

        allrolattn = None
        for k in range(len(ss2ds[stage])):
            regs = getattr(ss2ds[stage][k], "__data__")
            attn, H, W = cls.attnmap_mamba(regs, mode=mode1, ret=mode, absnorm=absnorm, verbose=verbose, scale=scale)
            L = H * W
            assert attn.shape == (L, L)
            assert attn.max() <= 1
            assert attn.min() >= 0
            rolattn = 0.5 * (attn.cpu() + torch.eye(L))
            rolattn = rolattn / rolattn.sum(-1)
            allrolattn = (rolattn @ allrolattn) if allrolattn is not None else rolattn
        return allrolattn
        

# used for test throughput
class Throughput:
    # default no amp in testing tp
    # copied from swin_transformer
    @staticmethod
    @torch.no_grad()
    def throughput(data_loader, model, logger=logging):
        model.eval()
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                model(images)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @staticmethod
    @torch.no_grad()
    def throughputamp(data_loader, model, logger=logging):
        model.eval()

        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                with torch.cuda.amp.autocast():
                    model(images)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                with torch.cuda.amp.autocast():
                    model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @staticmethod
    def testfwdbwd(data_loader, model, logger, amp=True):
        model.cuda().train()
        criterion = torch.nn.CrossEntropyLoss()

        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(images)
                    loss = criterion(out, targets)
                    loss.backward()
            torch.cuda.synchronize()
            logger.info(f"testfwdbwd averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(images)
                    loss = criterion(out, targets)
                    loss.backward()
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} testfwdbwd {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @classmethod    
    def testall(cls, model, dataloader=None, data_path="", img_size=224, _batch_size=128, with_flops=True, inference_only=False):
        from fvcore.nn import parameter_count
        torch.cuda.empty_cache()
        model.cuda().eval()
        if with_flops:
            try:
                FLOPs.fvcore_flop_count(model, input_shape=(3, img_size, img_size), show_arch=False)
            except Exception as e:
                print("ERROR:", e, flush=True)
        print(parameter_count(model)[""], sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
        if dataloader is None:
            dataloader = get_val_dataloader(
                batch_size=_batch_size, 
                root=os.path.join(os.path.abspath(data_path), "val"),
                img_size=img_size,
            )
        cls.throughput(data_loader=dataloader, model=model, logger=logging)
        if inference_only:
            return
        PASS = False
        batch_size = _batch_size
        while (not PASS) and (batch_size > 0):
            try:
                _dataloader = get_val_dataloader(
                    batch_size=batch_size, 
                    root=os.path.join(os.path.abspath(data_path), "val"),
                    img_size=img_size,
                )
                cls.testfwdbwd(data_loader=_dataloader, model=model, logger=logging)
                cls.testfwdbwd(data_loader=_dataloader, model=model, logger=logging, amp=False)
                PASS = True
            except:
                batch_size = batch_size // 2
                print(f"batch_size {batch_size}", flush=True)



# used for extract features
class ExtractFeatures:
    @staticmethod
    def get_list_dataset(*args, **kwargs):
        class DatasetList:
            def __init__(self, batch_size=16, root="train/", img_size=224, weak_aug=False):
                self.batch_size = int(batch_size)
                transform, transform_waug = self.get_transform(img_size)
                self.transform = transform_waug if weak_aug else transform
                self.dataset = datasets.ImageFolder(root, transform=self.transform)
                
                self.num_data = int(len(self.dataset))
                self.num_batches = math.ceil(self.num_data / self.batch_size)
                print(f"weak aug: {weak_aug} =========================", flush=True)

            @staticmethod
            def get_transform(img_size=224):
                size = int((256 / 224) * img_size)
                transform = transforms.Compose([
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ])
                transform_waug = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ])
                return transform, transform_waug

            def __len__(self):
                return self.num_batches

            def __getitem__(self, idx):
                start = idx * self.batch_size
                end = min(start + self.batch_size, self.num_data)
                data = [self.dataset[i] for i in range(start, end)]
                images = torch.stack([img for img, tgt in data])
                targets = torch.stack([torch.tensor(tgt) for img, tgt in data])
                if len(images) < self.batch_size:
                    _images = torch.zeros((self.batch_size, *data[0][0].shape))
                    _targets = -1 * torch.ones((self.batch_size,))
                    _images[:len(images)] = images
                    _targets[:len(images)] = targets
                    return _images, _targets
                return images, targets
        return DatasetList(*args, **kwargs)

    @classmethod
    def extract_feature(
        cls,
        backbones=dict(), # dict(name=model)
        batch_size=16, 
        img_size=1024,  
        data_path="ImageNet_ILSVRC2012", 
        amp_disable=False, 
        dims=dict(),  # dict(name=dim)
        outdir=os.path.join(HOME, "ckpts/feats/unmerge/"),
        ranges=[0, 1000],
        train=True,
        aug=False,
    ):
        root = os.path.join(data_path, "./train") if train else os.path.join(data_path, "./val")
        datasetlist = cls.get_list_dataset(batch_size, root=root, img_size=img_size, weak_aug=aug)

        ranges = list(ranges)
        if ranges[1] <= 0:
            ranges[1] = len(datasetlist)
        ranges[1] = min(ranges[1], len(datasetlist))
        assert len(ranges) == 2 and ranges[1] > ranges[0], f"{ranges}"
        outbatches = ranges[1] - ranges[0]
        outdir = os.path.join(outdir, f"sz{img_size}_bs{batch_size}_range{ranges[0]}_{ranges[1]}" + ("" if train else "_val"))
        os.makedirs(outdir, exist_ok=True)
        backbones = {
            name: torch.nn.parallel.DistributedDataParallel(model.cuda().eval())
            for name, model in backbones.items()
        }
        feats = {
            name: torch.zeros((outbatches, batch_size, dim))
            for name, dim in dims.items()
        }
        all_targets = torch.zeros((outbatches, batch_size))

        print("=" * 50, flush=True)
        print(f"using backbones {backbones.keys()}", flush=True)
        print(f"batch_size {batch_size} img_size {img_size} ranges {ranges} max_range {0} {len(datasetlist)}", flush=True)

        with torch.no_grad():
            for i, idx in enumerate(tqdm.tqdm(range(ranges[0], ranges[1]))):
                images, targets = datasetlist[idx]
                images = images.cuda(non_blocking=True)
                all_targets[i] = targets.detach().cpu()
                for name, model in backbones.items():
                    with torch.cuda.amp.autocast(enabled=(not amp_disable)):
                        feats[name][i] = model(images).detach().cpu()
            
            for name, model in backbones.items():
                na = f"{name}_bs{batch_size}_sz{img_size}_obs{outbatches}_s{ranges[0]}_e{ranges[1]}.pth"
                torch.save(feats[name], open(os.path.join(outdir, na), "wb"))
            na = f"targets_bs{batch_size}_sz{img_size}_obs{outbatches}_s{ranges[0]}_e{ranges[1]}.pth"
            torch.save(all_targets, open(os.path.join(outdir, na), "wb"))

    @staticmethod
    def merge_feats(features=[], targets=[], length=1281167, save="/tmp/1.pth"):
        feats = [torch.load(open(f, "rb")) for f in features]
        tgts = [torch.load(open(f, "rb")) for f in targets]
        for i, (f, t) in enumerate(zip(feats, tgts)):
            assert f.shape[0:2] == t.shape[0:2], breakpoint()
        assert sum([t.shape[0] for t in tgts]) * tgts[0].shape[1] >= length
        print(features, targets, flush=True)
        feats = torch.cat(feats, dim=0).view(-1, feats[0].shape[-1])
        tgts = torch.cat(tgts, dim=0).view(-1)
        if not (len(feats) == length):
            assert (feats[length:] == feats[length]).all() # input 0, models output same
            assert (feats[length] != feats[length - 1]).any()
            assert (tgts[length:] == -1).all()
            assert (tgts[:length] != -1).all()
        feats = feats[:length]
        tgts = tgts[:length]
        os.makedirs(os.path.dirname(save), exist_ok=True)
        assert not os.path.exists(save), f"file {save} exist"
        torch.save(dict(features=feats, targets=tgts), open(save, "wb"))


# used for print flops
class FLOPs:
    @staticmethod
    def register_supported_ops():
        build = import_abspy("models", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"))
        selective_scan_flop_jit: Callable = build.vmamba.selective_scan_flop_jit
        # flops_selective_scan_fn: Callable = build.vmamba.flops_selective_scan_fn
        # flops_selective_scan_ref: Callable = build.vmamba.flops_selective_scan_ref 
        def causal_conv_1d_jit(inputs, outputs):
            """
            https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
            x: (batch, dim, seqlen) weight: (dim, width) bias: (dim,) out: (batch, dim, seqlen)
            out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
            """
            from fvcore.nn.jit_handles import conv_flop_jit
            return conv_flop_jit(inputs, outputs)
        
        supported_ops={
            "aten::gelu": None, # as relu is in _IGNORED_OPS
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # =====================================================
            # for mamba-ssm
            "prim::PythonOp.CausalConv1dFn": causal_conv_1d_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
            # =====================================================
            # for VMamba
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            # "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCuda": selective_scan_flop_jit,
            # =====================================================
            # "aten::scaled_dot_product_attention": ...
        }
        return supported_ops

    @staticmethod
    def check_operations(model: nn.Module, inputs=None, input_shape=(3, 224, 224)):
        from fvcore.nn.jit_analysis import _get_scoped_trace_graph, _named_modules_with_dup, Counter, JitModelAnalysis
        
        if inputs is None:
            assert input_shape is not None
            if len(input_shape) == 1:
                input_shape = (1, 3, input_shape[0], input_shape[0])
            elif len(input_shape) == 2:
                input_shape = (1, 3, *input_shape)
            elif len(input_shape) == 3:
                input_shape = (1, *input_shape)
            else:
                assert len(input_shape) == 4

            inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)


        model.eval()

        flop_counter = JitModelAnalysis(model, inputs)
        flop_counter._ignored_ops = set()
        flop_counter._op_handles = dict()
        assert flop_counter.total() == 0 # make sure no operations supported
        print(flop_counter.unsupported_ops(), flush=True)
        print(f"supported ops {flop_counter._op_handles}; ignore ops {flop_counter._ignored_ops};", flush=True)

    @classmethod
    def fvcore_flop_count(cls, model: nn.Module, inputs=None, input_shape=(3, 224, 224), show_table=False, show_arch=False, verbose=True):
        supported_ops = cls.register_supported_ops()
        from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
        from fvcore.nn.flop_count import flop_count, FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS
        from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
        from fvcore.nn.jit_analysis import _IGNORED_OPS
        from fvcore.nn.jit_handles import get_shape, addmm_flop_jit
        
        if inputs is None:
            assert input_shape is not None
            if len(input_shape) == 1:
                input_shape = (1, 3, input_shape[0], input_shape[0])
            elif len(input_shape) == 2:
                input_shape = (1, 3, *input_shape)
            elif len(input_shape) == 3:
                input_shape = (1, *input_shape)
            else:
                assert len(input_shape) == 4

            inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)


        model.eval()

        Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)
        
        flops_table = flop_count_table(
            flops = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
            max_depth=100,
            activations=None,
            show_param_shapes=True,
        )

        flops_str = flop_count_str(
            flops = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
            activations=None,
        )

        if show_arch:
            print(flops_str)

        if show_table:
            print(flops_table)

        params = fvcore_parameter_count(model)[""]
        flops = sum(Gflops.values())

        if verbose:
            print(Gflops.items())
            print("GFlops: ", flops, "Params: ", params, flush=True)
        
        return params, flops

    # equals with fvcore_flop_count
    @classmethod
    def mmengine_flop_count(cls, model: nn.Module = None, input_shape = (3, 224, 224), show_table=False, show_arch=False, _get_model_complexity_info=False):
        supported_ops = cls.register_supported_ops()
        from mmengine.analysis.print_helper import is_tuple_of, FlopAnalyzer, ActivationAnalyzer, parameter_count, _format_size, complexity_stats_table, complexity_stats_str
        from mmengine.analysis.jit_analysis import _IGNORED_OPS
        from mmengine.analysis.complexity_analysis import _DEFAULT_SUPPORTED_FLOP_OPS, _DEFAULT_SUPPORTED_ACT_OPS
        from mmengine.analysis import get_model_complexity_info as mm_get_model_complexity_info
        
        # modified from mmengine.analysis
        def get_model_complexity_info(
            model: nn.Module,
            input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...],
                            None] = None,
            inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...],
                        None] = None,
            show_table: bool = True,
            show_arch: bool = True,
        ):
            if input_shape is None and inputs is None:
                raise ValueError('One of "input_shape" and "inputs" should be set.')
            elif input_shape is not None and inputs is not None:
                raise ValueError('"input_shape" and "inputs" cannot be both set.')

            if inputs is None:
                device = next(model.parameters()).device
                if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
                    inputs = (torch.randn(1, *input_shape).to(device), )
                elif is_tuple_of(input_shape, tuple) and all([
                        is_tuple_of(one_input_shape, int)
                        for one_input_shape in input_shape  # type: ignore
                ]):  # tuple of tuple of int, construct multiple tensors
                    inputs = tuple([
                        torch.randn(1, *one_input_shape).to(device)
                        for one_input_shape in input_shape  # type: ignore
                    ])
                else:
                    raise ValueError(
                        '"input_shape" should be either a `tuple of int` (to construct'
                        'one input tensor) or a `tuple of tuple of int` (to construct'
                        'multiple input tensors).')

            flop_handler = FlopAnalyzer(model, inputs).set_op_handle(**supported_ops)
            # activation_handler = ActivationAnalyzer(model, inputs)

            flops = flop_handler.total()
            # activations = activation_handler.total()
            params = parameter_count(model)['']

            flops_str = _format_size(flops)
            # activations_str = _format_size(activations)
            params_str = _format_size(params)

            if show_table:
                complexity_table = complexity_stats_table(
                    flops=flop_handler,
                    # activations=activation_handler,
                    show_param_shapes=True,
                )
                complexity_table = '\n' + complexity_table
            else:
                complexity_table = ''

            if show_arch:
                complexity_arch = complexity_stats_str(
                    flops=flop_handler,
                    # activations=activation_handler,
                )
                complexity_arch = '\n' + complexity_arch
            else:
                complexity_arch = ''

            return {
                'flops': flops,
                'flops_str': flops_str,
                # 'activations': activations,
                # 'activations_str': activations_str,
                'params': params,
                'params_str': params_str,
                'out_table': complexity_table,
                'out_arch': complexity_arch
            }
        
        if _get_model_complexity_info:
            return get_model_complexity_info

        model.eval()
        analysis_results = get_model_complexity_info(
            model,
            input_shape,
            show_table=show_table,
            show_arch=show_arch,
        )
        flops = analysis_results['flops_str']
        params = analysis_results['params_str']
        # activations = analysis_results['activations_str']
        out_table = analysis_results['out_table']
        out_arch = analysis_results['out_arch']
        
        if show_arch:
            print(out_arch)
        
        if show_table:
            print(out_table)
        
        split_line = '=' * 30
        print(f'{split_line}\nInput shape: {input_shape}\t'
            f'Flops: {flops}\tParams: {params}\t'
            #   f'Activation: {activations}\n{split_line}'
        , flush=True)
        # print('!!!Only the backbone network is counted in FLOPs analysis.')
        # print('!!!Please be cautious if you use the results in papers. '
        #       'You may need to check if all ops are supported and verify that the '
        #       'flops computation is correct.')

    @classmethod
    def mmdet_flops(cls, config=None, extra_config=None):
        from mmengine.config import Config
        from mmengine.runner import Runner
        import numpy as np
        import os

        cfg = Config.fromfile(config)
        if "model" in cfg:
            if "pretrained" in cfg["model"]:
                cfg["model"].pop("pretrained")
        if extra_config is not None:
            new_cfg = Config.fromfile(extra_config)
            new_cfg["model"] = cfg["model"]
            cfg = new_cfg
        cfg["work_dir"] = "/tmp"
        cfg["default_scope"] = "mmdet"
        runner = Runner.from_cfg(cfg)
        model = runner.model.cuda()
        get_model_complexity_info = cls.mmengine_flop_count(_get_model_complexity_info=True)
        
        if True:
            oridir = os.getcwd()
            os.chdir(os.path.join(os.path.dirname(__file__), "../detection"))
            data_loader = runner.val_dataloader
            num_images = 100
            mean_flops = []
            for idx, data_batch in enumerate(data_loader):
                if idx == num_images:
                    break
                data = model.data_preprocessor(data_batch)
                model.forward = partial(model.forward, data_samples=data['data_samples'])
                # out = get_model_complexity_info(model, inputs=data['inputs'])
                out = get_model_complexity_info(model, input_shape=(3, 1280, 800))
                params = out['params_str']
                mean_flops.append(out['flops'])
            mean_flops = np.average(np.array(mean_flops))
            print(params, mean_flops)
            os.chdir(oridir)

    @classmethod
    def mmseg_flops(cls, config=None, input_shape=(3, 512, 2048)):
        from mmengine.config import Config
        from mmengine.runner import Runner

        cfg = Config.fromfile(config)
        cfg["work_dir"] = "/tmp"
        cfg["default_scope"] = "mmseg"
        runner = Runner.from_cfg(cfg)
        model = runner.model.cuda()
        
        cls.fvcore_flop_count(model, input_shape=input_shape)

