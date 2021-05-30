import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import timm
import logging
import numpy as np
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import f1_score
from .loss import FocalLoss, ArcFaceLoss, LabelSmoothingCrossEntropy
from .loss import report
from torch.utils.data.sampler import WeightedRandomSampler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

class Trainer:
    def __init__(self, cfg, model, df_len):
        self.cfg = cfg
        self.model = model
        
        
        logging.basicConfig(filename=cfg.log_dir,
                            filemode='w',
                            level=logging.INFO, 
                            format='%(asctime)s  |  %(message)s')
        
        
        model_summary = summary(model)
        logging.info(f"====Model Summary====\n{model_summary}\n")
        logging.info(f"====Config Summary====\n{str(self.cfg)}")
        
        self.scoring_dict = {"preds": [], "labels": []}
        self.prediction_by_idx = np.array([0 for _ in range(self.cfg.cls_num+1)] * df_len)
        self.prediction_by_idx = np.reshape(self.prediction_by_idx, (-1, 19)).astype(np.float32)

    def reset_scoring_dict(self):
        self.scoring_dict = {"preds": [], "labels": []}
    
    def set_criterion(self, trn_df, val_df):
        logging.info("Criterion Setting...")
        
        if self.cfg.cls_weight:
            val_counts = trn_df.label.value_counts().sort_index().values
            trn_cls_weights = 1/np.log1p(val_counts)
            trn_cls_weights = (trn_cls_weights / trn_cls_weights.sum()) * self.cfg.cls_num
            trn_cls_weights = torch.tensor(trn_cls_weights, dtype=torch.float32).to(self.cfg.device)
            
            val_counts = val_df.label.value_counts().sort_index().values
            val_cls_weights = 1/np.log1p(val_counts)
            val_cls_weights = (val_cls_weights / val_cls_weights.sum()) * self.cfg.cls_num
            val_cls_weights = torch.tensor(val_cls_weights, dtype=torch.float32).to(self.cfg.device)
        else:
            trn_cls_weights = None
            val_cls_weights = None
        
        if self.cfg.crit == "focal":
            self.trn_crit = FocalLoss(type=self.cfg.focal_type, weight=trn_cls_weights)
            self.val_crit = FocalLoss(type=self.cfg.focal_type, weight=val_cls_weights)
        elif self.cfg.crit == 'arcface':
            self.trn_crit = ArcFaceLoss(self.cfg, weight=trn_cls_weights)
            self.val_crit = ArcFaceLoss(self.cfg, weight=val_cls_weights)
        elif self.cfg.crit == 'bce':
            self.trn_crit = nn.CrossEntropyLoss(weight=trn_cls_weights)
            self.val_crit = nn.CrossEntropyLoss(weight=val_cls_weights)
        elif self.cfg.crit == 'smoothing':
            self.trn_crit = LabelSmoothingCrossEntropy(weight=trn_cls_weights)
            self.val_crit = LabelSmoothingCrossEntropy(weight=val_cls_weights)
            
        logging.info("Done.\n")
        
        
    def set_loader(self, trn_ds, val_ds, uda_ds, batch):
        if self.cfg.weighed_sampler:
            targets = torch.from_numpy(trn_ds.data.label.values)
            class_sample_count = torch.tensor(
                [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
            weight = 1. / class_sample_count.float()
            samples_weight = torch.tensor([weight[t] for t in targets])
            sampler = WeightedRandomSampler(samples_weight.double(), len(samples_weight))
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        logging.info("Dataloader Setting...")
        self.trn_dl = DataLoader(
            trn_ds,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=4,
            sampler=sampler,
            pin_memory = True
        )
        
        self.val_dl = DataLoader(
            val_ds,
            batch_size=batch,
            shuffle=False,
            num_workers=4,
            pin_memory = True
        )
        
        self.uda_dl = DataLoader(
            uda_ds,
            batch_size=batch,
            shuffle=True,
            num_workers=4,
            pin_memory = True
        )
        self.uda_iter = iter(self.uda_dl)
        
        logging.info("Done.\n")
    
        
    def set_sched(self):
        logging.info("Scheduler Setting...")
        # self.sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optim, self.cfg.sched_T_0, eta_min=self.cfg.eta_min
        # )
        
        if self.cfg.sched_type == "cosine":
            self.sched = optim.lr_scheduler.CosineAnnealingLR(
                self.optim, self.cfg.sched_T_0, eta_min=self.cfg.eta_min
            )
        elif self.cfg.sched_type == 'onecycle':
            self.sched = optim.lr_scheduler.OneCycleLR(self.optim,
                                                       max_lr=self.cfg.lr, 
                                                       steps_per_epoch=len(self.trn_dl), 
                                                       epochs=self.cfg.t_epoch)
        elif self.cfg.sched_type == 'plateau':
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.optim,
                                                              mode='min', 
                                                              factor=0.5, 
                                                              patience=2)
            
        logging.info("Done.\n")
    
    
    def set_optim(self):
        logging.info("Optimizer Setting...")
        
        if self.cfg.optim == "Adam":
            self.optim = optim.Adam(params=self.model.parameters(),
                                    weight_decay=self.cfg.weight_decay,
                                    lr=self.cfg.lr)
        elif self.cfg.optim == "SGD":
            self.optim = optim.SGD(params=self.model.parameters(),
                                    weight_decay=self.cfg.weight_decay,
                                    momentum=0.9, nesterov=True,
                                    lr=self.cfg.lr)
        
        logging.info("Done.\n")
        
    
    def trn_step(self, epoch, sample_l, sample_u, scaler=None):
        self.optim.zero_grad()
        
        images_l = sample_l['image'].to(self.cfg.device)
        images_o = sample_u["image_ori"].to(self.cfg.device)
        images_a = sample_u["image_aug"].to(self.cfg.device)
        labels = sample_l['label'].to(self.cfg.device)
        
        batch_s = images_l.size(0)
        images_t = torch.cat([images_l, images_o, images_a])
        if scaler is not None:
            with autocast():
                logits_t = self.model(images_t)
        
                logits_l = logits_t[:batch_s]
                logits_o, logits_a = logits_t[batch_s:].chunk(2)
                del logits_t
                
                preds_o = F.softmax(logits_o, dim=-1).detach()
                preds_a = F.log_softmax(logits_a, dim=-1)
                kl_loss = F.kl_div(preds_a, preds_o, reduction='none')
                kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))
                    
                l_loss = self.trn_crit(logits_l, labels)
        
                if self.cfg.ratio_mode == 'constant':
                    t_loss = l_loss + self.cfg.ratio * torch.mean(kl_loss)
                elif self.cfg.ratio_mode == "gradual":
                    t_loss = epoch/self.cfg.t_epoch * self.cfg.ratio * torch.mean(kl_loss) + l_loss 
        
            scaler.scale(t_loss).backward()
            # clipping point -> batchnorm을 대체하는 역할 AGC
            scaler.unscale_(self.optim)
            if self.cfg.clipping:
                timm.utils.adaptive_clip_grad(self.model.parameters())
            scaler.step(self.optim)
            scaler.update()
        else:
            logits_t = self.model(images_t)
            logits_l = logits_t[:batch_s]
            logits_o, logits_a = logits_t[batch_s:].chunk(2)
            del logits_t
            
            preds_o = F.softmax(logits_o, dim=-1).detach()
            preds_a = F.log_softmax(logits_a, dim=-1)
            kl_loss = F.kl_div(preds_a, preds_o, reduction='none')
            kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))
                
            l_loss = self.trn_crit(logits_l, labels)
    
            if self.cfg.ratio_mode == 'constant':
                t_loss = l_loss + self.cfg.ratio * kl_loss
            elif self.cfg.ratio_mode == "gradual":
                t_loss = epoch/self.cfg.t_epoch * self.cfg.ratio * kl_loss + l_loss 
                
            t_loss.backward()
            
            if self.cfg.clipping:
                    timm.utils.adaptive_clip_grad(self.model.parameters())
            
            self.optim.step()
        
        batch_acc = self.accuracy(logits_l, labels)
        batch_f1  = self.f1_score(logits_l, labels)
        result = {
            'l_loss': l_loss,
            't_loss': t_loss,
            'kl_loss': kl_loss,
            'batch_acc': batch_acc,
            'batch_f1' : batch_f1
        }
        return result        
        
    
    def val_step(self, sample):
        images = sample['image'].to(self.cfg.device)
        labels = sample['label'].to(self.cfg.device)
        
        logits = self.model(images)
        loss = self.val_crit(logits, labels)
        
        batch_acc = self.accuracy(logits, labels)
        batch_f1  = self.f1_score(logits, labels)
        result = {
            't_loss': loss,
            'batch_acc': batch_acc,
            'batch_f1' : batch_f1
        }
        return result   
    
    
    def f1_score(self, logits, targs):
        _, preds = torch.max(logits, -1)
        preds = preds.detach().cpu().numpy()
        targs = targs.detach().cpu().numpy()
        
        self.scoring_dict["labels"].append(list(targs))
        self.scoring_dict["preds"].append(list(preds))
        return f1_score(preds, targs, average='macro')
        
        
    def accuracy(self, logits, targs):
        _, pred = torch.max(logits, -1)
        pred = pred.t()
        correct = pred.eq(targs).detach().cpu().numpy()
        return np.mean(correct)
    
    
    def train_on_epoch(self, fold, epoch):
        logging.info(f'Fold-[{fold}] ==> Training on epoch{epoch}...')
        
        trn_tloss_list = []
        trn_lloss_list = []
        trn_kloss_list = []
        trn_acc_list = [] 
        trn_f1_list = []
        self.model.train()
        
        if self.cfg.mixed_precision:
            scaler = GradScaler()
        else:
            scaler = None
        
        self.reset_scoring_dict()
        with tqdm(self.trn_dl, total=len(self.trn_dl), unit="batch") as train_bar:
            for batch, sample in enumerate(train_bar):
                train_bar.set_description(f"Fold-[{fold}|{self.cfg.n_fold}] ==> Train Epoch [{str(epoch).zfill(len(str(self.cfg.t_epoch)))}|{self.cfg.t_epoch}]")
                
                try:
                    sample_u = next(self.uda_iter)
                except StopIteration:
                    self.uda_iter = iter(self.uda_dl)
                    sample_u = next(self.uda_iter)
                
                result = self.trn_step(epoch, sample, sample_u, scaler)
                batch_f1  = result['batch_f1']
                batch_acc = result['batch_acc']
                t_loss = result['t_loss']
                l_loss = result['l_loss']
                k_loss = result['kl_loss']
                
                if not torch.isfinite(t_loss):
                    raise ValueError('WARNING: non-finite loss, ending training ')
                
                trn_f1_list.append(batch_f1)
                trn_acc_list.append(batch_acc)
                trn_tloss_list.append(t_loss.item())
                trn_lloss_list.append(l_loss.item())
                trn_kloss_list.append(k_loss.item())
                trn_f1 = np.mean(trn_f1_list)
                trn_acc = np.mean(trn_acc_list)
                trn_tloss = np.mean(trn_tloss_list)
                trn_lloss = np.mean(trn_lloss_list)
                trn_kloss = np.mean(trn_kloss_list)
                if batch % self.cfg.log_interval == 0 or batch == len(self.trn_dl)-1:
                    logging.info(f"Fold-[{fold}] ==> <Train> Epoch: [{str(epoch).zfill(len(str(self.cfg.t_epoch)))}|{str(self.cfg.t_epoch)}]  Batch: [{str(batch).zfill(len(str(len(self.trn_dl))))}|{len(self.trn_dl)}]\t Train Acc: {trn_acc}\t Train F1: {trn_f1}\t Train Loss: {trn_tloss:.6f} => [l_loss: {trn_lloss:.6f} | kl_loss: {trn_kloss:.6f}]")

                train_bar.set_postfix(train_t_loss=trn_tloss, train_l_loss=trn_lloss, train_kl_loss=trn_kloss, train_acc=trn_acc, train_f1=trn_f1)
                
                if self.cfg.sched_type == "onecycle":
                    self.sched.step()            
        
        if self.cfg.sched_type == "cosine": self.sched.step()
        reports = report(self.scoring_dict["preds"], self.scoring_dict["labels"])
        logging.info(f"Fold-[{fold}] ==> <Train> Epoch: [{str(epoch).zfill(len(str(self.cfg.t_epoch)))}|{str(self.cfg.t_epoch)}] REPOST\n{reports}\n")
    
    
    def valid_on_epoch(self, fold, epoch):
        logging.info(f'Fold-[{fold}] ==>  Validation on epoch{epoch}...')
        
        val_loss_list = []
        val_acc_list = []
        val_f1_list = []
        self.model.eval()
        
        self.reset_scoring_dict()
        with torch.no_grad():
            with tqdm(self.val_dl, total=len(self.val_dl), unit="batch") as valid_bar:
                for batch, sample in enumerate(valid_bar):
                    valid_bar.set_description(f"Fold-[{fold}|{self.cfg.n_fold}] ==> Valid Epoch [{str(epoch).zfill(len(str(self.cfg.t_epoch)))}|{self.cfg.t_epoch}]")
                    
                    result = self.val_step(sample)
                    batch_f1  = result['batch_f1']
                    batch_acc = result['batch_acc']
                    loss = result['t_loss']
                    
                    val_f1_list.append(batch_f1)
                    val_acc_list.append(batch_acc)
                    val_loss_list.append(loss.item())
                    val_f1 = np.mean(val_f1_list)
                    val_acc = np.mean(val_acc_list)
                    val_loss = np.mean(val_loss_list)
                    if batch % self.cfg.log_interval == 0 or batch == len(self.val_dl)-1:
                        logging.info(f"Fold-[{fold}] ==> <Valid> Epoch: [{str(epoch).zfill(len(str(self.cfg.t_epoch)))}|{str(self.cfg.t_epoch)}]  Batch: [{str(batch).zfill(len(str(len(self.val_dl))))}|{len(self.val_dl)}]\t Valid Acc: {val_acc}\t Valid F1: {val_f1}\t Valid Loss: {val_loss}")

                    valid_bar.set_postfix(valid_loss=val_loss,valid_acc=val_acc, valid_f1=val_f1)
        
        if self.cfg.sched_type == "plateau": self.sched.step(val_loss)
        reports = report(self.scoring_dict["preds"], self.scoring_dict["labels"])
        logging.info(f"Fold-[{fold}] ==> <Valid> Epoch: [{str(epoch).zfill(len(str(self.cfg.t_epoch)))}|{str(self.cfg.t_epoch)}] REPOST\n{reports}\n")
        return val_loss, val_acc, val_f1
                
                
    def save(self, fold, epoch, val_result, best_result):
        try:
            state_dict = self.model.module.state_dict()
        except Exception as e:
            state_dict = self.model.state_dict()
        
        name = ['acc', 'f1']
        index = [1, 2]
        for i, name in zip(index, name):
            if val_result[i] > best_result[i]:
                status = {
                    'epoch': epoch,
                    'loss': val_result[i],
                    'state_dict': state_dict,
                    'optim': self.optim.state_dict(),
                }
                
                filename = f"Fold-{fold}_" + f"best_{name}.pth.tar"
                torch.save(status, os.path.join(self.cfg.checkpoint, filename))
                logging.info(
                    f'Save Best Model ==> Epoch:{epoch}  |  Best {name}: {best_result[i]:.6f}  ----->  {val_result[i]:.6f}'
                )
                
                best_result[i] = val_result[i]
        
        if val_result[0] < best_result[0]:
            name = "loss"
            status = {
                        'epoch': epoch,
                        'loss': val_result[0],
                        'state_dict': state_dict,
                        'optim': self.optim.state_dict(),
                    }
                    
            filename = f"Fold-{fold}_" + f"best_loss.pth.tar"
            torch.save(status, os.path.join(self.cfg.checkpoint, filename))
            logging.info(
                f'Save Best Model ==> Epoch:{epoch}  |  Best {name}: {best_result[0]:.6f}  ----->  {val_result[0]:.6f}'
            )
            
            best_result[0] = val_result[0]
        return best_result

    def load(self, ckpt):
        if os.path.exists(ckpt):
            save_dict = torch.load(ckpt)
            state_dict = save_dict["state_dict"]

            try:
                self.model.module.load_state_dict(state_dict)
            except AttributeError as e:
                self.model.load_state_dict(state_dict)

            logging.info(f"{ckpt} model loaded.")
            print(f"{ckpt} model loaded.")
        else:
            logging.info("File doesn't exsist. Please check the directory.")
            logging.info("Start training with initailized model.\n")