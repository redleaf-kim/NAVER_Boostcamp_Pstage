import os
import torch
import random
import argparse
import numpy as np
import pandas as pd

from src.models import *
from functools import partial
from src.dataset import BoostcampDataset, prepare, UnlabeledDataset, UDATestDataset
from src.earlyStop import EarlyStopping
from sklearn.model_selection import StratifiedKFold, KFold


def seed_everything(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    import imgaug
    imgaug.random.seed(seed)



def main():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--seed', default=43, type=int, help='Reproduction Seed')
    # train
    parser.add_argument('--main_dir', default='/opt/ml', type=str, help='Main Code Directory')
    parser.add_argument('--n_fold', default=3, type=int, help='KFold Ensemble')
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--s_fold', default=1, type=int)
    parser.add_argument('--s_epoch', default=1, type=int)
    parser.add_argument('--t_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--eta_min', default=7e-5, type=float)
    parser.add_argument('--T_max', default=0, type=int)
    parser.add_argument('--decay', default=0, type=float)
    parser.add_argument('--mixed_precision', default=0, type=int)
    parser.add_argument('--weighted_sampler', default=0, type=int)
    parser.add_argument('--gridshuffle', default=1, type=int)
    parser.add_argument('--sched_type', default="cosine", type=str)
    parser.add_argument('--nosiy_elimination', default=0, type=int)
    
    # dataset
    parser.add_argument('--split', default='label', type=str)
    parser.add_argument('--postfix', type=str, required=True)
    parser.add_argument('--age_filter', default=59, type=int)
    
    # model
    parser.add_argument('--model_type', default='tf_efficientnet_b0_ns', type=str)
    parser.add_argument('--embed_size', default=512, type=int)
    parser.add_argument('--pool', default='gem', type=str)
    parser.add_argument('--neck', default='option-D', type=str)
    parser.add_argument('--multi_dropout', default=1, type=int)
    
    #criterion
    parser.add_argument('--cls_weight', default=1, type=int)
    parser.add_argument('--crit', default='arcface', type=str)
    parser.add_argument('--arcface_crit', default='focal', type=str)
    parser.add_argument('--focal_type', default='bce', type=str)
    
    #uda
    parser.add_argument('--uda', default=1, type=int)
    parser.add_argument('--ratio', default=5.0, type=float)
    parser.add_argument('--ratio_mode', default="constant", type=str)
    parser.add_argument('--uda_type', default='additional', type=str)

    #pseudo label
    parser.add_argument('--pseudo_label', default=0, type=int)
    parser.add_argument('--pseudo_label_path', default='tf_efficientnet_b3_ns_v5.csv', type=str)

    args = parser.parse_args() 
    
    if args.uda == 1: 
        from src.uda_trainer import Trainer
        from src.configs.uda_config import Config
    else: 
        from src.trainer import Trainer
        from src.configs.config import Config
        
    seed_everything(args.seed)
    cfg = Config(args, main_dir=args.main_dir)
    t_df = prepare(cfg, age_filter=args.age_filter)
    

    print("Training Starts.")
    if args.split == "id":
        def split_ids(value, indexs):
            if value in indexs: return True
            else: return False
            
        person_ids = [i for i in range(2700)]
        kfold = KFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)
        for fold_idx, (trn_idx, val_idx) in enumerate(kfold.split(person_ids), 1):
            if str(fold_idx) != str(args.s_fold) and str(args.s_fold) != '0': continue
        
            torch.cuda.empty_cache()
            
            t_df['trn'] = t_df['ids'].apply(partial(split_ids, indexs=trn_idx))
            trn_df = t_df.loc[t_df['trn']==True]
            val_df = t_df.loc[t_df['trn']==False]
            trn_df = trn_df.iloc[:, :2]
            val_df = val_df.iloc[:, :2]

            if args.pseudo_label:
                pseudo_df = pd.read_csv(cfg.pseudo_label_data)
                pseudo_df.columns = ["image", "label"]
                trn_df = pd.concat([trn_df, pseudo_df], axis=0)

            trn_ds = BoostcampDataset(cfg, trn_df, cfg.trn_tfms, train=True)
            val_ds = BoostcampDataset(cfg, val_df, cfg.val_tfms, train=False)
            
            model = Net(cfg)
            model.to(cfg.device)
            
            trainer = Trainer(cfg, model, df_len=len(t_df))
            if args.uda == 1 and args.uda_type=='additional':
                uda_ds = UnlabeledDataset(cfg)
                trainer.set_loader(trn_ds, val_ds, uda_ds, batch=args.batch_size)
            elif args.uda == 1 and args.uda_type == 'test':
                uda_ds = UDATestDataset(cfg)
                trainer.set_loader(trn_ds, val_ds, uda_ds, batch=args.batch_size)
            else:
                trainer.set_loader(trn_ds, val_ds, batch=args.batch_size)
            trainer.set_criterion(trn_df, val_df)
            trainer.set_optim()
            trainer.set_sched()
            
            
            if cfg.weight_path is not None:
                trainer.load(cfg.weight_path)
            
            early_stop = EarlyStopping(patience=5)
            best_result = [float("INF"), 0, 0]
            for epoch in range(args.s_epoch, args.t_epoch+1):
                trainer.train_on_epoch(fold_idx, epoch)
            
                # valid
                val_result = trainer.valid_on_epoch(fold_idx, epoch)
                best_result = trainer.save(fold_idx, epoch, val_result, best_result)
                if early_stop(val_result[0]):
                    break
                
    elif args.split == 'label':
        kfold = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)
        for fold_idx, (trn_idx, val_idx) in enumerate(kfold.split(t_df, t_df.label.values), 1):
            if str(fold_idx) != str(args.s_fold) and str(args.s_fold) != '0': continue
            
            torch.cuda.empty_cache()

            trn_df = t_df.iloc[trn_idx, :2]
            val_df = t_df.iloc[val_idx, :2]
            if args.pseudo_label:
                pseudo_df = pd.read_csv(cfg.pseudo_label_data)
                pseudo_df.columns = ["image", "label"]
                trn_df = pd.concat([trn_df, pseudo_df], axis=0)

            
            trn_ds = BoostcampDataset(cfg, trn_df, cfg.trn_tfms)
            val_ds = BoostcampDataset(cfg, val_df, cfg.val_tfms)

            # model = BasicNet(cfg)
            model = Net(cfg)
            model.to(cfg.device)
            
            trainer = Trainer(cfg, model, len(t_df))
            if args.uda == 1 and args.uda_type == 'additional':
                uda_ds = UnlabeledDataset(cfg)
                trainer.set_loader(trn_ds, val_ds, uda_ds, batch=args.batch_size)
            elif args.uda == 1 and args.uda_type == 'test':
                uda_ds = UDATestDataset(cfg)
                trainer.set_loader(trn_ds, val_ds, uda_ds, batch=args.batch_size)
            else:
                trainer.set_loader(trn_ds, val_ds, batch=args.batch_size)
            trainer.set_criterion(trn_df, val_df)
            trainer.set_optim()
            trainer.set_sched()
            
            
            
            if cfg.weight_path is not None:
                trainer.load(cfg.weight_path)
            
            best_result = [float("INF"), 0, 0]
            early_stop = EarlyStopping(patience=5)
            for epoch in range(args.s_epoch, args.t_epoch+1):
                trainer.train_on_epoch(fold_idx, epoch)
            
                # valid
                val_result = trainer.valid_on_epoch(fold_idx, epoch)
                best_result = trainer.save(fold_idx, epoch, val_result, best_result)
                
                if early_stop(val_result[0]):
                    break
                
    elif args.split == 'gender_ages':
        kfold = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)
        for fold_idx, (trn_idx, val_idx) in enumerate(kfold.split(t_df, t_df.gender_ages.values), 1):
            if str(fold_idx) != str(args.s_fold) and str(args.s_fold) != '0': continue
            
            torch.cuda.empty_cache()
            
            trn_df = t_df.iloc[trn_idx, :2]
            val_df = t_df.iloc[val_idx, :2]
            if args.pseudo_label:
                pseudo_df = pd.read_csv(cfg.pseudo_label_data)
                pseudo_df.columns = ["image", "label"]

                def pathfix(path):
                    return os.path.join('/opt/ml/input/data/eval/cropped_images', path)
                # print(pseudo_df.label.value_counts().sort_index().values)
                pseudo_df['image'] = pseudo_df['image'].apply(pathfix)
                trn_df = pd.concat([trn_df, pseudo_df], axis=0)

            trn_ds = BoostcampDataset(cfg, trn_df, cfg.trn_tfms)
            val_ds = BoostcampDataset(cfg, val_df, cfg.val_tfms)

            model = Net(cfg)
            model.to(cfg.device)
            
            trainer = Trainer(cfg, model, len(t_df))
            if args.uda == 1 and args.uda_type == 'additional':
                uda_ds = UnlabeledDataset(cfg)
                trainer.set_loader(trn_ds, val_ds, uda_ds, batch=args.batch_size)
            elif args.uda == 1 and args.uda_type == 'test':
                uda_ds = UDATestDataset(cfg)
                trainer.set_loader(trn_ds, val_ds, uda_ds, batch=args.batch_size)
            else:
                trainer.set_loader(trn_ds, val_ds, batch=args.batch_size)
            trainer.set_criterion(trn_df, val_df)
            trainer.set_optim()
            trainer.set_sched()
            
            if cfg.weight_path is not None:
                trainer.load(cfg.weight_path)

            best_result = [float("INF"), 0, 0]
            early_stop = EarlyStopping(patience=5)
            for epoch in range(args.s_epoch, args.t_epoch+1):
                trainer.train_on_epoch(fold_idx, epoch)
            
                # valid
                val_result = trainer.valid_on_epoch(fold_idx, epoch)
                best_result = trainer.save(fold_idx, epoch, val_result, best_result)
                
                if early_stop(val_result[0]):
                    break
                    
if __name__ == "__main__":
    main()