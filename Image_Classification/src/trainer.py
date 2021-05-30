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
        
        
    def set_loader(self, trn_ds, val_ds, batch):
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
        
    
    def step(self, sample, scaler=None, valid=False):
        images = sample['image'].to(self.cfg.device)
        labels = sample['label'].to(self.cfg.device)
        
        if scaler is not None:
            with autocast():
                logits = self.model(images)
                if not valid:
                    loss = self.trn_crit(logits, labels)
                else:
                    loss = self.val_crit(logits, labels)
            
            if not valid:                
                scaler.scale(loss).backward()
                # clipping point -> batchnorm을 대체하는 역할 AGC
                scaler.unscale_(self.optim)
                if self.cfg.clipping:
                    timm.utils.adaptive_clip_grad(self.model.parameters())
                scaler.step(self.optim)
                scaler.update()
        else:
            logits = self.model(images)
            if not valid:
                loss = self.trn_crit(logits, labels)
                
                self.optim.zero_grad()
                loss.backward()
                
                if self.cfg.clipping:
                    timm.utils.adaptive_clip_grad(self.model.parameters())
                
                self.optim.step()
            else:
                loss = self.val_crit(logits, labels)
            
            
            if self.cfg.nosiy_elimination:
                logit_preds = -F.log_softmax(logits, dim=-1)
                indexs = sample['idx'].detach().cpu().numpy()
                self.prediction_by_idx[indexs][:, :-1] += self.prediction_by_idx[indexs][:, :-1] * 0.2
                self.prediction_by_idx[indexs][:, :-1] += logit_preds
                self.prediction_by_idx[indexs][:, -1] = labels.detach().cpu().numpy()
            
        batch_acc = self.accuracy(logits, labels)
        batch_f1  = self.f1_score(logits, labels)
        result = {
            'logit': logits,
            'loss': loss,
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
        
        trn_loss_list = []
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
                
                result = self.step(sample, scaler)
                batch_f1  = result['batch_f1']
                batch_acc = result['batch_acc']
                loss = result['loss']
                
                if not torch.isfinite(loss):
                    print(loss, sample, result['logit'], sample['image'].shape, sample['label'].shape)
                    raise ValueError('WARNING: non-finite loss, ending training ')
                
                trn_f1_list.append(batch_f1)
                trn_acc_list.append(batch_acc)
                trn_loss_list.append(loss.item())
                trn_f1 = np.mean(trn_f1_list)
                trn_acc = np.mean(trn_acc_list)
                trn_loss = np.mean(trn_loss_list)
                if batch % self.cfg.log_interval == 0 or batch == len(self.trn_dl)-1:
                    logging.info(f"Fold-[{fold}] ==> <Train> Epoch: [{str(epoch).zfill(len(str(self.cfg.t_epoch)))}|{str(self.cfg.t_epoch)}]  Batch: [{str(batch).zfill(len(str(len(self.trn_dl))))}|{len(self.trn_dl)}]\t Train Acc: {trn_acc}\t Train F1: {trn_f1}\t Train Loss: {trn_loss}")

                train_bar.set_postfix(train_loss=trn_loss, train_acc=trn_acc, train_f1=trn_f1)
                
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
        
        if self.cfg.mixed_precision: scaler = GradScaler()
        else: scaler = None
            
        self.reset_scoring_dict()
        with torch.no_grad():
            with tqdm(self.val_dl, total=len(self.val_dl), unit="batch") as valid_bar:
                for batch, sample in enumerate(valid_bar):
                    valid_bar.set_description(f"Fold-[{fold}|{self.cfg.n_fold}] ==> Valid Epoch [{str(epoch).zfill(len(str(self.cfg.t_epoch)))}|{self.cfg.t_epoch}]")
                    
                    
                    result = self.step(sample, scaler, valid=True)
                    batch_f1  = result['batch_f1']
                    batch_acc = result['batch_acc']
                    loss = result['loss']
                    
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

            epoch = save_dict["epoch"]
            state_dict =  save_dict["state_dict"]
            optim_dict =  save_dict["optim"]

            self.optim.load_state_dict(optim_dict)
            try:
                self.model.module.load_state_dict(state_dict)
            except AttributeError as e:
                self.model.load_state_dict(state_dict)
            self.cfg.s_epoch = epoch

            logging.info(f"{ckpt} model loaded.")
            logging.info(f"Epoch re-start at {epoch}.\n")
        else:
            logging.info("File doesn't exsist. Please check the directory.")
            logging.info("Start training with initailized model.\n")