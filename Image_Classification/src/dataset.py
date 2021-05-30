import os
from albumentations.augmentations.transforms import HorizontalFlip
import cv2
import numpy as np
import pandas as pd

import torch
import albumentations as A
from torch.utils.data import Dataset



def prepare(cfg, age_filter=59, train=True):
    ids = []
    labels = []
    gender_ages = []
    image_paths = []
    
    
    meta = pd.read_csv(cfg.meta_dir)
    
    if train:
        def age(value):
            if value < 30: return 0
            elif 30 <= value < age_filter: return 1
            else: return 2
            
        def gender(value):
            if value == 'female': return 1
            else: return 0
            
        def gender_age(gender, age):
            return gender * 3 + age

        meta['age'] = meta['age'].apply(age)
        meta['gender'] = meta['gender'].apply(gender)
        meta['gender_age'] = meta.apply(lambda x: gender_age(x.gender, x.age), axis=1)
        
        
        for i in range(len(meta)):
            id, gender, _, age, path, gender_age = meta.iloc[i, :]
            
            if str(id) in ['006364', '006363', '006362', '006361', '006360', '006359']: gender = 0
            elif str(id) in ['001498-1', '004432']: gender = 1
            
            images = os.listdir(cfg.image_folder/path)
            images = [image for image in images if not image.startswith('.')]
            
            for image in images:
                image_path = cfg.image_folder / path / image
                
                if image.startswith('incorrect'): mask_group = 1
                elif image.startswith('normal'): mask_group = 2
                else: mask_group = 0
                
                label = 6 * mask_group + 3 * gender + age
                ids.append(i)
                labels.append(label)  
                gender_ages.append(gender_age)
                image_paths.append(image_path)
                
        
        data =  pd.DataFrame({
            'image': image_paths,
            'label': labels,
            'ids': ids,
            'gender_ages': gender_ages
        })
                        
        
    else:
        for i in range(len(meta)):
            path, _ = meta.iloc[i, :]
            image_path = cfg.image_folder / path
            
            image_paths.append(image_path)
        
        data = pd.DataFrame({
            'image': image_paths
        })
        
    return data
                
                
class BoostcampDataset(Dataset):
    def __init__(self, cfg, data, transform=None, train=True):
        self.cfg = cfg
        self.train = train
        self.data = data
        self.transform = transform
        if self.cfg.gridshuffle:
            self.gridshuffle = A.OneOf([
                A.RandomGridShuffle((2, 2), p=1.0),
                # A.RandomGridShuffle((4, 2), p=1.0),
            ], p=0.5)
        else:
            self.gridshuffle = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data.iloc[idx, :]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
            # if self.train and self.gridshuffle is not None:
            # if self.train and label not in [i for i in range(6)] and self.gridshuffle is not None:# --> best result
            if self.train and label not in [i for i in range(12)] and self.gridshuffle is not None:
                image = self.gridshuffle(image=image)['image']
        
        sample = {
            'idx': torch.tensor(idx, dtype=torch.long),
            'image': torch.from_numpy(image).permute(2, 1, 0).float(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        return sample
    

class BoostcampTTATestDataset(Dataset):
    def __init__(self, cfg, data, transform=None, tta_tfms=None):
        self.cfg = cfg
        self.data = data
        self.transform = transform
        self.tta_tfms = tta_tfms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images = [image]
        images.extend([
            tta_tfm(image=image)['image'] for tta_tfm in self.tta_tfms
        ])

        images = [
            torch.from_numpy(self.transform(image=img)['image']).permute(2, 1, 0).float() for img in images
        ]

        sample = {
            'image': images
        }
        return sample


    
class BoostcampTestDataset(Dataset):
    def __init__(self, cfg, data, transform=None):
        self.cfg = cfg
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            if self.transform:
                image = self.transform(image=image)['image']
        except Exception as e:
            print(image_path)

        sample = {
            'image': torch.from_numpy(image).permute(2, 1, 0).float(),
        }

        return sample


class PseudoDataset(Dataset):
    def __init__(self, cfg, tta_tfms=None):
        self.cfg = cfg
        self.data = self.prepare()
        self.mean = self.cfg.mean
        self.std = self.cfg.std
        self.tta_tfms = tta_tfms

        self.normal_transform = A.Compose([
            A.Resize(384, 288, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(p=1.0, mean=self.mean, std=self.std)
        ])

    def prepare(self):
        data = []
        main_folder = self.cfg.data_dir / 'unlabeled'

        for p_folder in [main_folder / 'AFDB_face_dataset', main_folder / 'AFDB_masked_face_dataset']:
            for c_folder in os.listdir(p_folder):
                for file in os.listdir(p_folder / c_folder):
                    data.append(
                        str(p_folder / c_folder / file)
                    )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images = [image]
        images.extend([
            tta_tfm(image=image)['image'] for tta_tfm in self.tta_tfms
        ])

        images = [
            torch.from_numpy(self.normal_transform(image=img)['image']).permute(2, 1, 0).float() for img in images
        ]

        sample = {
            'path': img_path,
            'image': images
        }
        return sample


class UnlabeledDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = self.prepare()
        self.mean = self.cfg.mean
        self.std = self.cfg.std

        self.normal_transform = A.Compose([
            A.Resize(384, 288, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(p=1.0, mean=self.mean, std=self.std)
        ])

        self.augment_transform = A.Compose([
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

    def prepare(self):
        data = []
        main_folder = self.cfg.unlabeld_folder
        
        for p_folder in [main_folder/'AFDB_face_dataset', main_folder/'AFDB_masked_face_dataset']:
            for c_folder in os.listdir(p_folder):
                for file in os.listdir(p_folder/c_folder):
                    data.append(
                        str(p_folder/c_folder/file)
                    )
        return data
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        imgPath = self.data[index]
        
        image = cv2.imread(str(imgPath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_ori = self.normal_transform(image=image)['image']
        image_aug = self.augment_transform(image=image)['image']
        
        sample = {
            'image_ori': torch.from_numpy(image_ori).permute(2, 1, 0).float(),
            'image_aug': torch.from_numpy(image_aug).permute(2, 1, 0).float(),
        }
        
        return sample
    
    
    
class UDATestDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = self.prepare()
        self.mean = self.cfg.mean
        self.std = self.cfg.std

        self.normal_transform = A.Compose([
            A.Resize(384, 288, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(p=1.0, mean=self.mean, std=self.std)
        ])
        
        self.augment_transform = A.Compose([
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
    
    def prepare(self):
        data = []
        main_folder = self.cfg.data_dir / 'eval' / 'cropped_images'
        for file in os.listdir(str(main_folder)):
            if not file.startswith('.'):
                data.append(
                    str(main_folder/file)
                )
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_ori = self.normal_transform(image=image)['image']
        image_aug = self.augment_transform(image=image)['image']
        
        sample = {
            'image_ori': torch.from_numpy(image_ori).permute(2, 1, 0).float(),
            'image_aug': torch.from_numpy(image_aug).permute(2, 1, 0).float(),
        }
        
        return sample