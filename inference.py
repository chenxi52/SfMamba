import time
import random
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim
from models import build_model
from utils.logger import create_logger
from datautil.getdataloader import get_tgt_img_dataloader
from utils.dgutil import train_valid_target_eval_names_DA, img_param_init, eval_accuracy_gpu
from config import get_config
import os

def str2bool(v):
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
    parser.add_argument('--dataset', type=str, default='office-home', choices=['office','office-home','VISDA-C','domainnet126'])
    parser.add_argument('--source_env', type=int, nargs='+', default=0)
    parser.add_argument('--target_env', type=int, default=0)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.25)

    parser.add_argument('--dg_aug', type=str2bool, default=True, help='Use augmentation')
    parser.add_argument('--neck', type=str, default='bn_relu', choices=['','bn_relu','bn_relu_ln'])
    parser.add_argument('--sel', type=str2bool, default=True, help='selet clean samples and supervised contrastive learning')
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


def main(config,args):
    _, data_loader_val = get_tgt_img_dataloader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()

    ###### load  model ######
    model.layers.load_state_dict(torch.load(config.MODEL.TARGET_PATH+'/target_F.pt',weights_only=True), strict=False)
    model.patch_embed.load_state_dict(torch.load(config.MODEL.TARGET_PATH+'/target_F.pt',weights_only=True), strict=False)
    if args.neck!=''  and not (args.neck=='bn_relu' and config.MODEL.NECK_DEEP==0):
        model.fusion_blocks.load_state_dict(torch.load(config.MODEL.TARGET_PATH+'/target_B.pt',weights_only=True))
    model.classifier.load_state_dict(torch.load(config.MODEL.TARGET_PATH+'/target_C.pt',weights_only=True))

    logger.info("Start Inference")
    start_time = time.time()
    if config.dataset == 'VISDA-C':
        target_acc = validate(config, data_loader_val, model, flag=True)
        acc_te, acc_list = target_acc[0]
        logger.info(f"Accuracy of the network on the TARGET test images: {acc_te * 100:.1f}% \n acc_list: {acc_list}")
    else:
        target_acc = validate(config, data_loader_val, model, flag=False)
        logger.info(f"Accuracy of the network on the TARGET test images: {target_acc[0]*100:.1f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Inference time {}'.format(total_time_str))
    

@torch.no_grad()
def validate(config, data_loader, model, flag=False):
    acc_record = {}
    eval_name_dict = train_valid_target_eval_names_DA(config)
    acc_record['target'] = [eval_accuracy_gpu(model,
                                data_loader[eval_name_dict['target'][0]],flag, 
                                amp=config.AMP_ENABLE,
                                num_classes=config.MODEL.NUM_CLASSES)]
    return acc_record['target']

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

    config.defrost()
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    main(config,args)