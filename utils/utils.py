# --------------------------------------------------------
# Modified By Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from math import inf
import torch
import torch.distributed as dist
from timm.utils import ModelEma as ModelEma
import sys

def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, domain='target'):
    training_state = torch.load(config.OUTPUT + '/training_state.pt', map_location='cpu', weights_only=False)
    print('loading from epoch=',training_state['epoch'],'\n')

    optimizer.load_state_dict(training_state['optimizer'])  
    loss_scaler.load_state_dict(training_state['scaler'])
    lr_scheduler.load_state_dict(training_state['lr_scheduler'])
    config.defrost()
    config.TRAIN.START_EPOCH = training_state['epoch'] + 1
    config.freeze()
    ##### load model state #####
    model.layers.load_state_dict(torch.load(config.OUTPUT+f'/{domain}_F.pt',weights_only=True), strict=False)
    model.patch_embed.load_state_dict(torch.load(config.OUTPUT+f'/{domain}_F.pt',weights_only=True), strict=False)
    model.fusion_blocks.load_state_dict(torch.load(config.OUTPUT+f'/{domain}_B.pt',weights_only=True), strict=False)
    model.classifier.load_state_dict(torch.load(config.OUTPUT+f'/{domain}_C.pt',weights_only=True), strict=False)
    if training_state['epoch'] ==  config.TRAIN.EPOCHS-1 and not config.MODEL.RESUME:
        print(f"Training already finished at epoch {config.TRAIN.EPOCHS-1}, no need to resume.")
        sys.exit(0)
    del training_state
    torch.cuda.empty_cache()


def load_pretrained_ema(config, model, logger, model_ema: ModelEma=None, load_ema_separately=False):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if k in checkpoint['model'] and model_dict[k].shape != checkpoint['model'][k].shape:
            del checkpoint['model'][k]
    if 'head.weight' in checkpoint['model']:
        del checkpoint['model']['head.weight']
        del checkpoint['model']['head.bias']
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        key = "model_ema" if load_ema_separately else "model"
        if key in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint[key], strict=False)
            logger.warning(msg)
            logger.info(f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema")
        else:
            logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint_ema(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None, max_accuracy_ema=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    
    if model_ema is not None:
        save_state.update({'model_ema': model_ema.ema.state_dict(),
            'max_accuray_ema': max_accuracy_ema})

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pt')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    return checkpoints


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm
    
    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)