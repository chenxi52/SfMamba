import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import wandb
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim
from utils.optimizer import  check_keywords_in_name
from timm.utils import AverageMeter
from models import build_model
from utils.lr_scheduler import build_scheduler
from utils.logger import create_logger
from utils.utils import  NativeScalerWithGradNormCount
import torch.nn as nn
from utils.loss import myCriterion
from datautil.getdataloader import get_tgt_img_dataloader
from utils.dgutil import train_valid_target_eval_names_DA, img_param_init, eval_accuracy_gpu
from config import get_config
from scipy.spatial.distance import cdist
from utils.utils import load_checkpoint
from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
from torch.cuda.amp import autocast

@torch.no_grad()
def obtain_label(loader, model, not_cluster=False):
    start_test = True
    model.eval()
    num_steps = len(loader)
    print('start obtain label .....')
    with torch.no_grad():
        for step ,data in enumerate(loader):
            if step % 100 == 0:
                print(f'Obtaining label at Batch {step}/{num_steps}')
                torch.cuda.empty_cache()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs,feas = model(inputs, return_feat=True)
            feas = nn.AdaptiveAvgPool2d(1)(feas).flatten(1)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat([all_fea, feas.float().cpu()], 0)
                all_output = torch.cat([all_output, outputs.float().cpu()], 0)
                all_label = torch.cat([all_label, labels.float()], 0)
    all_output = nn.Softmax(dim=1)(all_output)  
    _, predict = torch.max(all_output, 1)
    
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    distance = 'cosine'
    all_fea = torch.cat([all_fea, torch.ones(all_fea.size(0), 1)], 1) 
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t() 
    if not_cluster:
        return predict.cuda(), all_fea[:, :-1].float().cuda()
    
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy() 
    initc = aff.transpose().dot(all_fea)  
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], distance) 
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label] 
        initc = aff.transpose().dot(all_fea) 
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = f'Pseudo Accuracy = {accuracy * 100:.2f}% -> {acc * 100:.2f}%'
    logger.info(log_str)    
    model.train()
    return torch.from_numpy(pred_label.astype('int')).cuda(), torch.from_numpy(all_fea[:, :-1]).cuda()


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
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', type=str2bool, default=0, help='resume from checkpoint')
    parser.add_argument('--accumulation_steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    parser.add_argument('--dataset', type=str, default='office-home', choices=['office','office-home','VISDA-C','domainnet126'])
    parser.add_argument('--target_env', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--output', type=str, default=None)

    # mine parameters 
    parser.add_argument('--par_ent', type=float, default=1.0)
    parser.add_argument('--par_cls', type=float, default=0.3)
    parser.add_argument('--par_st', type=float, default=1., help='par for soft target loss')
    parser.add_argument('--div_loss', type=str2bool, help='use div loss', default=True)
    parser.add_argument('--source_env', type=int, nargs='+', default=0)
    parser.add_argument('--issave', action='store_true', help='save the target model', default=False)
    parser.add_argument('--dg_aug', type=str2bool, default=True, help='Use augmentation')
    parser.add_argument('--reg_weight', action='store_true', help='Use reg_weight', default=False)
    parser.add_argument('--neck', type=str, default='bn_relu', choices=['','bn_relu','bn_relu_ln'])
    parser.add_argument('--pseu_target', type=str, default='cluster', choices=['None','cluster','target'])
    parser.add_argument('--add_contrast', action='store_true', help='add contrastive loss', default=False)
    parser.add_argument('--lr_factor1', type=float, default=0.1)
    parser.add_argument('--lr_factor2', type=float, default=1.)
    parser.add_argument('--bg_ratio', type=float, default=0.2)
    parser.add_argument('--su_cl_t', type=float, default=5., help='tem for supervised contrastive loss')
    parser.add_argument('--sel', type=str2bool, default=True, help='selet clean samples and supervised contrastive learning')
    parser.add_argument('--clean_ratio', type=float, default=0.6, help='ratio for clean samples')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--fusion_module', type=str, default='ssm')
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
    args.target_da = True
    args = img_param_init(args)
    config = get_config(args)
    return args, config


def build_optimizer(config, model, logger):
    logger.info(f"==============> building optimizer {config.TRAIN.OPTIMIZER.NAME}....................")
    skip = {}
    skip_keywords = {}
    for name, param in model.classifier.named_parameters():
        param.requires_grad = False

    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = set_weight_decay_lr(model, config.TRAIN.BASE_LR, 
                                     config.lr_factor1, config.lr_factor2, 
                                     skip, skip_keywords)

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


def set_weight_decay_lr(model, base_lr, layer_lr_factor=0.1, 
                        layer_lr_factor2=1., skip_list=(), 
                        skip_keywords=()):
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
        if not 'classifier' in name : ### for feature extractor and patch_embed
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                speci_lr_no_decay.append(param)
            else:
                speci_lr_has_decay.append(param)
    return [
        {'params': speci_lr_has_decay2, 'lr': base_lr * layer_lr_factor2, 'weight_decay': 1e-4},
        {'params': speci_lr_no_decay2, 'lr': base_lr * layer_lr_factor2, 'weight_decay': 0.0}, 
        {'params': speci_lr_has_decay, 'lr': base_lr * layer_lr_factor, 'weight_decay': 1e-4}, 
        {'params': speci_lr_no_decay, 'lr': base_lr * layer_lr_factor, 'weight_decay': 0.0} 
    ]


def main(config,args):
    data_loader_train, data_loader_val = get_tgt_img_dataloader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()

    optimizer = build_optimizer(config, model, logger)
    ###### load source model ######
    logger.info(f"Loading source model from {config.MODEL.SOURCE_F}")
    
    loss_scaler = NativeScalerWithGradNormCount()
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train[0]) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train[0]))

    if args.resume and os.path.exists(config.OUTPUT + '/training_state.pt'):
        load_checkpoint(config, model, optimizer, lr_scheduler,loss_scaler)
    else:
        ##### load source model.#####
        model.layers.load_state_dict(torch.load(config.MODEL.SOURCE_F,weights_only=True), strict=False)
        model.patch_embed.load_state_dict(torch.load(config.MODEL.SOURCE_F,weights_only=True), strict=False)
        if args.neck!=''  and not (args.neck=='bn_relu' and config.MODEL.NECK_DEEP==0):
            model.fusion_blocks.load_state_dict(torch.load(config.MODEL.SOURCE_B,weights_only=True))
        model.classifier.load_state_dict(torch.load(config.MODEL.SOURCE_C,weights_only=True))

    criterion = myCriterion(div=args.div_loss, par_ent=args.par_ent, par_cls = args.par_cls, par_st=args.par_st)
    if config.THROUGHPUT_MODE:
        throughput(data_loader_val[0], model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, args, model, criterion, data_loader_train, data_loader_val,
                            optimizer, epoch, lr_scheduler, loss_scaler)
        if config.dataset == 'VISDA-C':
            target_acc = validate(config, data_loader_val, model, flag=True)
            acc_te, acc_list = target_acc[0]
            logger.info(f"Accuracy of the network on the TARGET test images: {acc_te * 100:.1f}% \n acc_list: {acc_list}")
        else:
            target_acc = validate(config, data_loader_val, model, flag=False)
            logger.info(f"Accuracy of the network on the TARGET test images: {target_acc[0]*100:.1f}%")
        if args.wandb: wandb.log({'acc':target_acc})

        if args.issave or args.resume:
            def save_model(temp_model):
                F_state_dict = temp_model.patch_embed.state_dict()
                F_state_dict.update(temp_model.layers.state_dict())
                torch.save(F_state_dict, config.OUTPUT+'/target_F.pt')
                if args.neck!='' and not (args.neck=='bn_relu' and config.MODEL.NECK_DEEP==0):
                    B_state_dict = temp_model.fusion_blocks.state_dict()
                    torch.save(B_state_dict, config.OUTPUT+'/target_B.pt')
                save_state = {
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'scaler': loss_scaler.state_dict(),
                        'epoch': epoch,
                        'config': config}
                torch.save(save_state, config.OUTPUT+'/training_state.pt')
                torch.save(temp_model.classifier.state_dict(), config.OUTPUT+'/target_C.pt')
                save_model(model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    

def train_one_epoch(config, args, model, criterion, data_loader, 
                    data_loader_val, optimizer, epoch, 
                    lr_scheduler, loss_scaler):
    optimizer.zero_grad()
    model.train()

    num_steps = len(data_loader[0])
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    train_minibatches_iterator = zip(*data_loader)
    if args.par_cls >0:
        pseudo_label, temp = obtain_label(data_loader_val[0], model, 
                                not_cluster=(args.pseu_target!='cluster' and epoch==0)) 
    else: pseudo_label = None

    if args.sel:
        all_fea = temp
        selected_examples = pair_selection_gpu(config, 4, data_loader_val[0], 
                                            pseudo_label, args.num_classes, 
                                            cos_t=args.su_cl_t, knn_times=2,
                                            train_features=all_fea, 
                                            sel_ratio=args.clean_ratio)

    for batch_idx in range(num_steps):
        minibatches = [(data) for data in next(train_minibatches_iterator)]
        samples = torch.cat([data[0].float() for data in minibatches])
        idxs = torch.cat([data[2].long() for data in minibatches])
        samples = samples.cuda(non_blocking=True)
        pseudo_target = pseudo_label[idxs] if pseudo_label is not None else None

        data_time.update(time.time() - end)
        with autocast(enabled=config.AMP_ENABLE):
            if config.MODEL.TYPE=='resnet':
                outputs = model(samples)
            else:
                outputs = model(samples, y=pseudo_target, 
                                    scs=args.par_st>0)
            if args.par_st>0:
                assert outputs.shape[0] != samples.size(0)
                ori_pred, masked_pred = outputs.split([samples.size(0),samples.size(0)])
                outputs = ori_pred
            else:
                masked_pred = None

            selected_batch = selected_examples[idxs.long()].bool() if selected_examples is not None else None
            loss = criterion(outputs, pseudo_target, masked_pred, selected_batch)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(batch_idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (batch_idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + batch_idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        loss_meter.update(loss.item(), samples.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - batch_idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{batch_idx}/{num_steps}]\t'
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
    eval_name_dict = train_valid_target_eval_names_DA(config)
    acc_record['target'] = [eval_accuracy_gpu(model,
                                data_loader[eval_name_dict['target'][0]],flag, 
                                amp=config.AMP_ENABLE,
                                num_classes=config.MODEL.NUM_CLASSES)]
    return acc_record['target']


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for idx, data in enumerate(data_loader):
        images = data[0]
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        Gflops,flop_table = model.flops()
        params = fvcore_parameter_count(model)[""]
        logger.info('flops: %.2f G, params: %.2f M' % (Gflops , params / 1e6))
        logger.info(flop_table)
        return

def pair_selection_gpu(config, k_val, test_loader, labels, class_num, 
                        cos_t, knn_times, train_features, sel_ratio=0):
    '''
    k_val:  neighbors number of knn
    labels: pseudo-labels obtained from feature prototypes
    '''
    train_labels = labels.clone().cuda()
    discrepancy_measure = torch.zeros(len(test_loader.dataset), device='cuda')

    with torch.no_grad():
        full_dist = torch.mm(train_features, train_features.t()).fill_diagonal_(-1)

        for i in range(knn_times):
            print(f'starting the {i+1}st knn....')
            for batch_idx, data in enumerate(test_loader):
                index = data[2].long()
                current_batch_size = index.size(0) 

                if batch_idx % 100 == 0:
                    print(f' Batch {batch_idx}/{len(test_loader)}')
                    torch.cuda.empty_cache() 

                with autocast(enabled=config.AMP_ENABLE):
                    dist = full_dist[index]
                    yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True) 
                    del dist

                    retrieval = train_labels[yi]  # Direct indexing avoids expand() 
                    retrieval_one_hot = torch.zeros(
                        current_batch_size * k_val, 
                        class_num,
                        device='cuda'
                    )
                    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1) 
                    
                    yd_transform = torch.exp(yd /cos_t) 
                    probs = torch.sum(
                        retrieval_one_hot.view(current_batch_size, k_val, class_num) * 
                        yd_transform.view(current_batch_size, k_val, 1),
                        dim=1
                    )
                    probs_norm = probs / probs.sum(dim=1, keepdim=True)

                    prob_temp = probs_norm[torch.arange(current_batch_size), labels[index]]
                    prob_temp = torch.clamp(prob_temp, 1e-4, 1-1e-4)
                    discrepancy_measure[index] = -torch.log(prob_temp)
                    
                    new_labels = probs_norm.argmax(dim=1)
                    train_labels[index] = new_labels  # In-place update

    del train_labels, train_features, full_dist
    torch.cuda.empty_cache()
    print("Applying class balancing...")
    selected_examples = torch.zeros(len(test_loader.dataset), device='cuda')
    
    for i in range(class_num):
        class_mask = labels == i
        num_class = class_mask.sum().item()
        if num_class == 0:
            continue
            
        class_indices = class_mask.nonzero().squeeze(1)
        class_scores = discrepancy_measure[class_indices]
        k_selected = max(1, int(sel_ratio * num_class))
        _, top_indices = torch.topk(class_scores, k=k_selected, largest=False, sorted=True)
        
        selected_indices = class_indices[top_indices]
        selected_examples[selected_indices] = 1
        
    return selected_examples



if __name__ == '__main__':
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False  
    cudnn.deterministic = True  

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR 
    linear_scaled_min_lr = config.TRAIN.MIN_LR
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr 
    config.TRAIN.MIN_LR = linear_scaled_min_lr 
    config.freeze()

    # to make sure all the config.OUTPUT are the same
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    if args.wandb:
        config.defrost()
        wandb.login()
        wandb.init(project = 'SfMamba',
                    config = config,
                    group=args.output.split('/')[-1]+f'{args.bg_ratio}', )
        config.freeze()
        wandb.run.name = config.OUTPUT.split('/')[-1]

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    import threading
    print(f"Active threads: {threading.active_count()}")
    main(config,args)
    if args.wandb:
        wandb.finish()