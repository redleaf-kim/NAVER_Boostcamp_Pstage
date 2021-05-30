import os
import torch
from pathlib import Path
import albumentations as A

class Config:
    def __init__(self, args, main_dir='/opt/ml/', mode='train'):
        self.postfix = args.postfix
        self.main_dir = Path(main_dir)
        self.data_dir = self.main_dir / 'input/data'
        self.image_folder = self.data_dir / mode / 'cropped_images'
        self.meta_dir = self.data_dir / mode / str(mode + '.csv')
        
        self.gridshuffle = True if args.gridshuffle == 1 else False
        self.mixed_precision = True if args.mixed_precision == 1 else False
        self.n_fold = args.n_fold
        self.s_epoch = args.s_epoch
        self.t_epoch = args.t_epoch
        self.weight_path = None
        self.weighed_sampler = True if args.weighted_sampler == 1 else False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        self.nosiy_elimination = True if args.nosiy_elimination == 1 else False
        
        # criterion
        self.clipping = True if "nfnet" in args.model_type else False
        self.crit = args.crit
        self.arcface_crit = args.arcface_crit
        self.focal_type = args.focal_type
        self.cls_weight = True if args.cls_weight == 1 else False
        self.focal_gamma = 5.0
        
        # optimizer 
        self.optim = args.optim
        self.lr = args.lr
        self.weight_decay = args.decay
        
        # scheduler
        self.sched_type = args.sched_type
        self.sched_T_0 = args.T_max if args.T_max != 0 else self.t_epoch
        self.eta_min = args.eta_min
        
        # model
        self.cls_num = 18
        self.backbone_name = args.model_type
        self.checkpoint = self.main_dir / 'checkpoints' / str(self.backbone_name + "_" + args.postfix)
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint, exist_ok=True)
        
        self.backbone_pretrained = True if mode == 'train' else False
        self.embed_size = args.embed_size
        
        self.pool = args.pool
        self.p_trainable = True
        self.neck = args.neck
        
        self.multi_dropout = True if args.multi_dropout  == 1 else False
        self.multi_dropout_num = 16
        self.multi_dropout_prob = 0.2

        # pseudo label
        self.pseudo_label = True if args.pseudo_label == 1 else False
        self.pseudo_label_data = self.main_dir / 'submission' / args.pseudo_label_path
        
        
        # logging
        self.log_interval = 50
        self.log_dir = self.main_dir / 'logs' 
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.log_dir = self.log_dir / (self.backbone_name + "_" + args.postfix + '.txt')
        
        self.mean = [0.56019358, 0.52410121, 0.501457]
        self.std  = [0.23318603, 0.24300033, 0.24567522]
        
        # transforms
        self.trn_tfms = A.Compose([
            A.Resize(384, 288, p=1.0),
            A.HorizontalFlip(p=0.7),
            A.GaussNoise(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25, rotate_limit=20, p=0.6, border_mode=0),
           
            A.OneOf([
                A.CLAHE(p=0.5),
                A.Compose([
                    A.RandomBrightness(limit=0.5, p=0.6),
                    A.RandomContrast(limit=0.4, p=0.6),
                    A.RandomGamma(p=0.6),        
                ])    
            ], p=0.65),
            
            A.OneOf([
                A.HueSaturationValue(10, 20, 10, p=1.0),
                A.RGBShift(p=1.0),
                A.Emboss(p=1.0),
            ], p=0.5),
            
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.3, p=0.3),
            
            A.OneOf([
                A.Perspective(p=1.0, scale=(0.05, 0.1)),
                A.GridDistortion(p=1.0, distort_limit=0.25, border_mode=0),
                A.OpticalDistortion(p=1.0, shift_limit=0.1, distort_limit=0.1, border_mode=0)
            ], p=0.65),

            A.Normalize(p=1.0, mean=self.mean, std=self.std),
        ])
        
        self.val_tfms = A.Compose([
            A.Resize(384, 288),
            A.Normalize(p=1.0, mean=self.mean, std=self.std),
        ])
        
        
    def __str__(self):
        temp = ''
        for key, value in self.__dict__.items():
            temp += f"{key} ==> {value}\n"
        
        return temp


class InferConfig:
    def __init__(self, args, main_dir='/opt/ml/', mode='eval'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.postfix = args.postfix
        self.main_dir = Path(main_dir)
        self.data_dir = self.main_dir / 'input/data'
        self.submission_dir = self.main_dir / 'submission'
        if not os.path.exists((str(self.submission_dir))):
            os.makedirs(str(self.submission_dir), exist_ok=True)

        self.image_folder = self.data_dir / mode / 'cropped_images'
        self.meta_dir = self.data_dir / mode / 'info.csv'

        self.mean = [0.56019358, 0.52410121, 0.501457]
        self.std  = [0.23318603, 0.24300033, 0.24567522]
        
        self.cls_num = 18
        self.backbone_name = args.model_type
        self.submission_dir /= str(self.backbone_name + "_" + self.postfix + '.csv')
        self.ckpts = [
            # 'checkpoints/tf_efficientnet_b3_ns_v5/Fold-1_best_loss.pth.tar',
            # 'checkpoints/tf_efficientnet_b3_ns_v5/Fold-2_best_loss.pth.tar',
            # 'checkpoints/tf_efficientnet_b3_ns_v5/Fold-3_best_loss.pth.tar',
            # 'checkpoints/tf_efficientnet_b3_ns_v5/Fold-4_best_loss.pth.tar'
            'checkpoints/tf_efficientnet_b3_ns_v7_fold1/Fold-1_best_loss.pth.tar',
        ]
        self.ckpts = [self.main_dir / path for path in self.ckpts]
        
        self.backbone_pretrained = True if mode == 'train' else False
        self.embed_size = 512
        
        self.pool = 'gem'
        self.p_trainable = True
        self.neck = 'option-D'
        
        self.multi_dropout = False
        self.multi_dropout_num = 16
        self.multi_dropout_prob = 0.2
        
        
        self.infer_tfms = A.Compose([
            A.Resize(384, 288),
            A.Normalize(p=1.0, mean=self.mean, std=self.std),
        ])
        