import torch.nn.functional as F
import torch
import torch.nn as nn

class myCriterion(nn.Module):
    def __init__(self, div=False, par_ent=1., par_cls=1.0, par_adv=1.0, 
                 par_st=1.0,):
        super().__init__()
        self.par_ent = par_ent
        self.par_cls = par_cls
        self.par_st = par_st
        self.par_adv = par_adv
        self.div = div

    def entropy_loss(self, ul_y):
        p=F.softmax(ul_y,dim=1)
        loss = -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)
        if self.div:
            msoftmax = p.mean(dim=0)
            div_loss = -torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
            loss += div_loss
        return loss * self.par_ent

    def pseudo_sup_loss(self, ul_y, pseudo_label=None):
        if not pseudo_label is None:
            loss = nn.CrossEntropyLoss()(ul_y, pseudo_label) 
        else:
            loss = 0.
        return loss* self.par_cls
    
    def soft_target_loss(self, output, soft_target=None):
        if output is not None:
            loss = nn.KLDivLoss(reduction='batchmean')\
                (F.log_softmax(output,dim=1),F.softmax(soft_target,dim=1))
        else:
            loss = 0.
        return loss * self.par_st
    

    def forward(self, x, label, x2, selected_inbatch=None):
        if selected_inbatch is not None and selected_inbatch.numel():
            x_sel = x[selected_inbatch]
            label_sel = label[selected_inbatch]
            x2_sel = x2[selected_inbatch] if x2 is not None else None
        elif selected_inbatch is not None and selected_inbatch.numel() == 0:
            x_sel = None
            label_sel = None
            x2_sel = None
        else:
            x_sel = x
            label_sel = label
            x2_sel = x2
        
        loss = 0.
        if self.par_ent > 0:
            loss += self.entropy_loss(x)
        if self.par_st>0 :
            loss += self.soft_target_loss(x2_sel, soft_target = x_sel)
        if self.par_cls>0:
            loss += self.pseudo_sup_loss(x_sel, label_sel)
        return loss
    

class CrossEntropyLabelSmooth(nn.Module):
    """
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    
