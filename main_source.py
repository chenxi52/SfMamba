import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim

from timm.utils import AverageMeter
from utils.loss import CrossEntropyLabelSmooth
from utils.optimizer import  check_keywords_in_name

from models import build_model
from utils.lr_scheduler import build_scheduler
from utils.logger import create_logger
from utils.utils import NativeScalerWithGradNormCount, load_pretrained_ema,load_checkpoint

from timm.utils import ModelEma as ModelEma
from datautil.getdataloader import get_img_dataloader
from utils.dgutil import train_valid_target_eval_names, eval_accuracy, img_param_init
from config import get_config


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--test_batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', default='',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', type=str2bool, default=False, help='resume from checkpoint')
    parser.add_argument('--accumulation_steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--test_envs', type=int, nargs='+', default=0)
    parser.add_argument('--split_style', type=str, default='strat')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--dg_aug', action='store_true', help='Use augmentation', default=False)
    parser.add_argument('--neck', type=str, default='bn_relu', 
                        choices=['','bn','bn_relu','bn_ln_relu','bn_relu_ln','bottleneck','eca','se','cbam'])
    parser.add_argument('--lr_factor1', type=float, default=0.1)
    parser.add_argument('--lr_factor2', type=float, default=1.)
    parser.add_argument('--fusion_module', type=str, default='ssm', choices=['None','ssm','cbam','se','eca'])

    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    args, unparsed = parser.parse_known_args()
    args = img_param_init(args)
    config = get_config(args)
    return args, config


def set_weight_decay_lr(model, base_lr, layer_lr_factor=0.1, layer_lr_factor2=1., skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    speci_lr_has_decay = []
    speci_lr_no_decay = []
    speci_lr_has_decay2 = []
    speci_lr_no_decay2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
        if 'fusion_blocks' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                speci_lr_no_decay2.append(param)
            else:
                speci_lr_has_decay2.append(param)
            continue
        if not 'classifier' in name: ### layers/patch_embed
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                speci_lr_no_decay.append(param)
            else:
                speci_lr_has_decay.append(param)
        else:
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                no_decay.append(param)
            else:
                has_decay.append(param)
    return [
        {'params': has_decay, 'lr': base_lr, 'weight_decay': 1e-4}, 
        {'params': no_decay, 'lr': base_lr, 'weight_decay': 0.0},
        {'params': speci_lr_has_decay2, 'lr': base_lr * layer_lr_factor2, 'weight_decay': 1e-4}, 
        {'params': speci_lr_no_decay2, 'lr': base_lr * layer_lr_factor2, 'weight_decay': 0.0}, 
        {'params': speci_lr_has_decay, 'lr': base_lr * layer_lr_factor, 'weight_decay': 1e-4},  
        {'params': speci_lr_no_decay, 'lr': base_lr * layer_lr_factor, 'weight_decay': 0.0}  
    ]


def build_optimizer(config, model, logger, **kwargs):
    logger.info(f"==============> building optimizer {config.TRAIN.OPTIMIZER.NAME}....................")
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = set_weight_decay_lr(model, config.TRAIN.BASE_LR, config.lr_factor1, config.lr_factor2, skip, skip_keywords)
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        raise NotImplementedError
    return optimizer


def main(config,args):
    data_loader_train, data_loader_val = get_img_dataloader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    model_ema = None

    optimizer = build_optimizer(config, model, logger)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train[0]) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train[0]))
    criterion = CrossEntropyLabelSmooth(num_classes=config.MODEL.NUM_CLASSES)

    best_valid_acc, best_target_acc = 0, 0
    best_epoch, best_epoch_ema = 0, 0

    if config.MODEL.RESUME:
        load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, domain='source')
        if config.EVAL_MODE:
            train_acc, valid_acc, target_acc = validate(config, 
                            data_loader_val, model, 
                            flag=config.dataset=='VISDA-C')
            loginfo = ''
            for do in range(len(config.domains)-1):
                loginfo += f'{target_acc[do]*100:.1f}% | '
            logger.info(loginfo)
            return
    if config.MODEL.PRETRAINED and config.MODEL.PRETRAINED != '' and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model, logger, model_ema)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, loss_scaler, model_ema)
        def save_model(temp_model):
            F_state_dict = temp_model.patch_embed.state_dict()
            F_state_dict.update(temp_model.layers.state_dict())
            torch.save(F_state_dict, config.OUTPUT+'/source_F.pt')
            if args.neck!='' and not (args.neck=='bn_relu' and config.MODEL.NECK_DEEP==0):
                B_state_dict = temp_model.fusion_blocks.state_dict()
                torch.save(B_state_dict, config.OUTPUT+'/source_B.pt')
            save_state = {
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': loss_scaler.state_dict(),
                    'epoch': epoch,
                    'config': config}
            torch.save(save_state, config.OUTPUT+'/training_state.pt')
            torch.save(temp_model.classifier.state_dict(), config.OUTPUT+'/source_C.pt')
       
        train_acc, valid_acc, target_acc = validate(config, data_loader_val, 
                                            model, flag=config.dataset=='VISDA-C')
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            if config.MODEL.VSSM.POSEMBED:
                assert model.pos_embed is not None
            else:
                save_model(model)
        loginfo = f'Epoch: {epoch} Best epoch: {best_epoch} target accuracy:'
        if config.dataset == 'VISDA-C':
            loginfo += f'{target_acc[0][0]*100:.1f}% \n acc_list: {target_acc[0][1]}'
        else:
            for do in range(len(config.domains)-1):
                loginfo += f'{target_acc[do]*100:.1f}% | '
        logger.info(loginfo)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, loss_scaler, model_ema=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader[0])
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    train_minibatches_iterator = zip(*data_loader)
    for idx in range(num_steps):
        minibatches = [(data) for data in next(train_minibatches_iterator)]
        samples = torch.cat([data[0].float() for data in minibatches])
        targets = torch.cat([data[1].long() for data in minibatches])
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        data_time.update(time.time() - end)
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, flag=False):
    acc_record = {}
    eval_name_dict = train_valid_target_eval_names(config) 
    acc_type_list = ['train', 'valid', 'target'] 
    for item in acc_type_list:
        if item == 'target':
            acc_record[item] = [eval_accuracy(model, data_loader[eval_name_dict[item][i]], flag) 
                                        for i in range(len(eval_name_dict[item]))]
        else:
            ############# train/valid acc ##############
            if flag:
                acc_record[item] = np.mean(np.array(
                    [eval_accuracy(model, data_loader[i], flag)[0] for i in eval_name_dict[item]]))
            else:
                acc_record[item] = np.mean(np.array(
                    [eval_accuracy(model, data_loader[i], flag) for i in eval_name_dict[item]]))
    infos = f" * train {acc_record['train']:.4f} valid {acc_record['valid']:.4f} target: "
    domain_num = len(config.domains)
    if flag:
        infos += f"{acc_record['target'][0][0]:.4f} \n acc_list: {acc_record['target'][0][1]}"
    else:
        assert domain_num - 1 == len(eval_name_dict[item]), f"domain_num {domain_num} eval_name_dict {eval_name_dict[item]}"
        for i in range(len(eval_name_dict[item])):
            infos += f" {acc_record['target'][i]:.4f} | "
    logger.info(infos)
    return acc_record['train'], acc_record['valid'], acc_record['target']


if __name__ == '__main__':
    args, config = parse_option()
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False  
    cudnn.deterministic = True  

    linear_scaled_lr = config.TRAIN.BASE_LR 
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR
    linear_scaled_min_lr = config.TRAIN.MIN_LR 
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr 
    config.TRAIN.MIN_LR = linear_scaled_min_lr 
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))
    main(config,args)
