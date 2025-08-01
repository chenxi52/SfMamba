import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'office'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'vssm' ### resnet
# Model name
_C.MODEL.NAME = 'vssm_tiny_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 7
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# MMpretrain models for test
_C.MODEL.MMCKPT = False

# VSSM parameters
_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.PATCH_SIZE = 4
_C.MODEL.VSSM.IN_CHANS = 3
_C.MODEL.VSSM.DEPTHS = [2, 2, 9, 2]
_C.MODEL.VSSM.EMBED_DIM = 96
_C.MODEL.VSSM.SSM_D_STATE = 16
_C.MODEL.VSSM.SSM_RATIO = 2.0
_C.MODEL.VSSM.SSM_RANK_RATIO = 2.0
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = True
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v2"
_C.MODEL.VSSM.MLP_RATIO = 4.0
_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0
_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.NORM_LAYER = "ln"
_C.MODEL.VSSM.DOWNSAMPLE = "v2"
_C.MODEL.VSSM.PATCHEMBED = "v2"
_C.MODEL.VSSM.POSEMBED = False
_C.MODEL.VSSM.GMLP = False
_C.MODEL.FUSION_MODULE = 'ssm'
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 50
_C.TRAIN.WARMUP_EPOCHS = 1
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 4e-4
_C.TRAIN.WARMUP_LR = 3e-7
_C.TRAIN.MIN_LR = 4e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 0.
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'output/'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False

######### add ########
_C.TRAIN.VALID_RATE = 0.1
_C.MODEL.SOURCE_F = ''
_C.MODEL.SOURCE_C = ''
_C.MODEL.STATE = ''
_C.MODEL.SOURCE_DIR = 'None'
_C.target_da = False
_C.MODEL.EMBED_DIM = 768
_C.MODEL.NECK = ''
_C.MODEL.NECK_DEEP=2
_C.MODEL.VSSM.SSM_FORWARDTYPE_1D = 'v052d'
_C.MODEL.NECK_RES = False
_C.MODEL.VSSM.NECK_NORM_LAYER='LN2d'
_C.TRAIN.BASE_D_LR = 1e-4
_C.bg_ratio=0.2


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)
    def _check_args(name):
        # if hasattr(args, name) and eval(f'args.{name}'):
        #     return True
        # return False
        return getattr(args, name, False) 
    config.defrost()
    if _check_args('lr'):
        config.TRAIN.BASE_LR = args.lr
        config.TRAIN.MIN_LR = args.lr * 0.05
    if args.opts:
        config.merge_from_list(args.opts)
    
    config.DATA.DATA_PATH = config.DATA.DATA_PATH + config.DATA.DATASET + '/'
    # merge from specific arguments
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    
    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    config.DATA.BATCH_SIZE = config.DATA.BATCH_SIZE // 2
    config.steps_per_epoch = 200
    if _check_args('dataset'):
        config.dataset = args.dataset
        config.DATA.DATASET = args.dataset
    if _check_args('test_envs'):
        config.test_envs = args.test_envs

    if _check_args('domains'):
        config.domains = args.domains
    if _check_args('img_dataset'):
        config.img_dataset = args.img_dataset[args.dataset]
        config.domain_num = len(args.img_dataset[args.dataset])
    if _check_args('num_classes'):
        config.MODEL.NUM_CLASSES = args.num_classes
    if _check_args('split_style'):
        config.split_style = args.split_style

    if _check_args('d_lr'):
        config.TRAIN.BASE_D_LR = args.d_lr
    if _check_args('alpha'):
        config.alpha = args.alpha
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('test_batch_size'):
        config.DATA.TEST_BATCH_SIZE = args.test_batch_size
    #### output folder ####
    ######## mine ########
    if _check_args('source_env'):
        config.source_env = args.source_env
    if _check_args('target_env'):
        config.target_env = args.target_env
    if _check_args('target_da'):
        config.target_da = args.target_da
    if _check_args('neck'):
        config.MODEL.NECK = args.neck
        
    config.dg_aug = args.dg_aug
    if _check_args('bg_ratio'):
        config.bg_ratio = args.bg_ratio
    if _check_args('sel_ratio'):
        config.sel_ratio = args.sel_ratio
    if _check_args('gamma'):
        config.gamma = args.gamma
    if _check_args('lr_factor1'):
        config.lr_factor1 = args.lr_factor1
    if _check_args('lr_factor2'):
        config.lr_factor2 = args.lr_factor2 
    if _check_args('fusion_module'):
        config.MODEL.FUSION_MODULE = args.fusion_module
    if not config.target_da:
        train_env = (set(list(range(len(config.domains)))) - set(config.test_envs)).pop()
        config.OUTPUT = os.path.join(config.OUTPUT,config.dataset, config.domains[train_env])
    else:
        config.OUTPUT = os.path.join(config.OUTPUT,config.dataset,f'{config.domains[args.source_env[0]]}_{config.domains[config.target_env]}')
        config.MODEL.SOURCE_F = os.path.join(config.MODEL.SOURCE_DIR,config.dataset,config.domains[args.source_env[0]],'source_F.pt') 
        config.MODEL.SOURCE_C = os.path.join(config.MODEL.SOURCE_DIR,config.dataset,config.domains[args.source_env[0]],'source_C.pt')
        config.MODEL.SOURCE_B = os.path.join(config.MODEL.SOURCE_DIR,config.dataset,config.domains[args.source_env[0]],'source_B.pt')
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
