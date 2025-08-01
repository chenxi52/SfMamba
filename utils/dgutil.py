import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.cuda.amp import autocast


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0    
    '''t represent the index of the dataloader in eval_loader, e.g., eval_loader = [0, 1, 2, 0, 1, 2, 3], 
    where 0-4 is the proxy of the domian, the 4-th domain is the target domain. the first three 0 1 2 is the source domain, 
    while the latter 0 1 2 is the valid-set, and the last 3 is the target domain'''
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict

def train_valid_target_eval_names_DA(args):
    eval_name_dict = {'train': [], 'target': []}
    '''t represent the index of the dataloader in eval_loader, e.g., eval_loader = [0, 1, 2, 0, 1, 2, 3], 
    where 0-4 is the proxy of the domian, the 4-th domain is the target domain. the first three 0 1 2 is the source domain, 
    while the latter 0 1 2 is the valid-set, and the last 3 is the target domain'''
    eval_name_dict['target'].append(0)
    return eval_name_dict

def eval_accuracy(model, loader, flag=False, amp=False, **kwargs):
    model.eval()
    start_test = True
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda(non_blocking=True)
            y = data[1]
            with autocast(enabled=amp):
                p = model(x, **kwargs)
                if flag:
                    if start_test:
                        all_output = p.cpu()
                        all_label = y
                        start_test = False
                    else:
                        all_output = torch.cat([all_output, p.cpu()], 0)
                        all_label = torch.cat([all_label, y], 0)
                else: 
                    _, predict = torch.max(p, 1)
                    correct += (predict == y.cuda()).sum().item()
                    total += y.size(0)
    model.train()

    if flag:
        _, predict = torch.max(all_output, 1)
        matrix = confusion_matrix(all_label, torch.squeeze(predict))
        acc = matrix.diagonal() / matrix.sum(axis=1)
        aacc = acc.mean()
        aa = [str(np.round(i, 4)) for i in acc]
        acc_str = ' '.join(aa)
        return aacc, acc_str
    else:
        accuracy = correct / total
        return accuracy 


def eval_accuracy_gpu(model, loader, flag=False, amp=False, **kwargs):
    model.eval()
    device = next(model.parameters()).device  
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        if flag:
            # Pre-allocate GPU tensors for full output collection
            num_samples = len(loader.dataset)
            num_classes = model.fc.out_features if hasattr(model, 'fc') else kwargs.get('num_classes', 10)
            all_output = torch.empty((num_samples, num_classes), device=device)
            all_label = torch.empty(num_samples, dtype=torch.long, device=device)
            
            start_idx = 0
            for data in loader:
                x = data[0].to(device, non_blocking=True)
                y = data[1].to(device, non_blocking=True)
                batch_size = x.size(0)
                
                with autocast(enabled=amp):
                    p = model(x, **kwargs)
                
                end_idx = start_idx + batch_size
                all_output[start_idx:end_idx] = p
                all_label[start_idx:end_idx] = y
                start_idx = end_idx
            
            # Compute accuracy metrics on GPU
            _, predict = torch.max(all_output, 1)
            correct = (predict == all_label).sum().item()
            total = all_label.size(0)
            
            # Move to CPU only for final confusion matrix calculation
            matrix = confusion_matrix(all_label.cpu().numpy(), 
                                    predict.cpu().numpy())
            acc = matrix.diagonal() / matrix.sum(axis=1)
            aacc = acc.mean()
            aa = [str(np.round(i, 4)) for i in acc]
            acc_str = ' '.join(aa)
            
            model.train()
            return aacc, acc_str
            
        else:
            # Standard accuracy calculation entirely on GPU
            for data in loader:
                x = data[0].to(device, non_blocking=True)
                y = data[1].to(device, non_blocking=True)
                
                with autocast(enabled=amp):
                    p = model(x, **kwargs)
                
                _, predict = torch.max(p, 1)
                correct += (predict == y).sum().item()
                total += y.size(0)
            
            model.train()
            return correct / total
    

def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    elif dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'VISDA-C':
        domains = ['train', 'validation']
    elif dataset == 'domainnet':
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    elif dataset == 'domainnet126':
        domains = ['clipart', 'painting', 'real', 'sketch']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office-home': ['Art', 'Clipart', 'Product', 'RealWorld'], 
        'office': ['amazon', 'dslr', 'webcam'], 
        'VISDA-C': ['train', 'validation'], 
        'domainnet126': ['clipart', 'painting', 'real', 'sketch'],
        'domainnet': ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    }
    args.input_shape = (3, 224, 224)
    if args.dataset == 'office-home':
        args.num_classes = 65
    elif args.dataset == 'office':
        args.num_classes = 31
    elif args.dataset == 'VISDA-C':
        args.num_classes = 12
    elif args.dataset == 'domainnet':
        args.num_classes = 345
    elif args.dataset == 'domainnet126':
        args.num_classes = 126
    return args