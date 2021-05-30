import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from src.models import *
from src.configs.config import InferConfig
from src.dataset import BoostcampTestDataset, BoostcampTTATestDataset, prepare

import torch.nn.functional as F
from torch.utils.data import DataLoader


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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--postfix', required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--tta', default=0, type=int)
    
    
    args = parser.parse_args() 
    seed_everything(args.seed)
    
    cfg = InferConfig(args)
    tta_infer = True if args.tta == 1 else False
    if tta_infer:
        print("TTA Inference")
        tta_tfms = [
            # A.CLAHE(clip_limit=2.0, p=1.0), --> 넣어도 같은 결과나옴
            A.HorizontalFlip(p=1.0),
        ]
    else:
        tta_tfms = None

    infer_tfms = cfg.infer_tfms
    infer_df = prepare(cfg, train=False)
    if tta_infer: infer_ds = BoostcampTTATestDataset(cfg, infer_df, infer_tfms, tta_tfms)
    else: infer_ds = BoostcampTestDataset(cfg, infer_df, infer_tfms)
    infer_dl = DataLoader(
        infer_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True
    )


    models = []
    for i in range(len(cfg.ckpts)):
        model = Net(cfg)
        model = model.to(cfg.device)
        save_dict = torch.load(cfg.ckpts[i])
        print(f"Epoch: {save_dict['epoch']}")
        print(f"Loss : {save_dict['loss']}")
        state_dict = save_dict["state_dict"]
        model.load_state_dict(state_dict)
        models.append(model)

    print(f"Total {len(models)} models loaded.")

    if tta_infer:
        predictions = []
        with torch.no_grad():
            for sample in tqdm(infer_dl, total=len(infer_dl)):
                images = sample['image']
                
                pred = 0
                for image in images:
                    for model in models:
                        model.eval()
                        pred = model(image.to(cfg.device))
                        pred += F.log_softmax(pred, dim=-1)
                    
                _, pred = torch.max(pred/(len(models)), -1)
                predictions.extend(pred.detach().cpu().numpy())        
        
    else: 
        predictions = []
        with torch.no_grad():
            for sample in tqdm(infer_dl, total=len(infer_dl)):
                images = sample['image'].to(cfg.device)
                
                pred = 0
                for model in models:
                    model.eval()
                    pred = model(images)
                    pred += F.log_softmax(pred, dim=-1)
                    
                _, pred = torch.max(pred/(len(models)), -1)
                predictions.extend(pred.detach().cpu().numpy())
            
    submission = pd.read_csv(cfg.meta_dir)
    submission['ans'] = predictions
    submission.to_csv(cfg.submission_dir, index=False)
    
    print("Inference Done.")
        
    
if __name__ == "__main__":
    main()