import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
from utils.augmentations import apply_augmentations
from utils.tokenizer import LaneTokenizer
from utils.clip import clip_annotation



class TuSimpleDataset(Dataset):
    def __init__(self, root_dir, split='train', nbins=1000, format_type='anchor', image_size=(320, 800), config=None):
        self.root_dir = root_dir
        self.split = split
        self.nbins = nbins
        self.format_type = format_type
        self.image_size = image_size  # (height, width)
        self.tokenizer = LaneTokenizer(nbins=self.nbins)
        self.aug_config = config.get('augmentation', {}) if config else {}
        print(f" Augmentation config loaded: {self.aug_config}")

        self.samples = []

        #  Prepare label files
        if split == 'train':
            label_files = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
        elif split == 'test':
            label_files = ['test_label.json']
        else:
            raise ValueError(f"Unsupported split: {split}")

        #  Read all samples
        for label_file in label_files:
            label_path = os.path.join(root_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))

        # Define augmentations
        self.transform = self.get_transforms(split)

    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.root_dir, sample['raw_file'])
        # Inside __getitem__ method

        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # original before resize
        
        # Apply annotation conversion (rescale points)
        annotation = self._convert_annotation(
            sample,
            original_size=original_size,
            target_size=(self.image_size[1], self.image_size[0])
        )
        
        # Resize image before augmentation (because we augment in resized space)
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        
        # Apply augmentations
        image, annotation = apply_augmentations(image, annotation, self.aug_config)
        
        # Tokenize
        input_seq, target_seq = self.tokenizer.encode(
            annotation,
            (self.image_size[1], self.image_size[0]),
            format_type=self.format_type
        )
        
        # Final transforms
        image = T.ToTensor()(image)
        image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        return {
            'image': image,
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target_seq': torch.tensor(target_seq, dtype=torch.long),
            'raw_file': sample['raw_file'],
            'annotation': annotation  # for test_tokenizer visual debugging
        }


    def get_transforms(self, split):
        return T.Compose([])  # No transforms at all


    def _convert_annotation(self, sample, original_size, target_size):
        lanes = []
        h_samples = sample['h_samples']
        img_width, img_height = original_size
        new_width, new_height = target_size
    
        x_scale = new_width / img_width
        y_scale = new_height / img_height
    
        for lane_points in sample['lanes']:
            points = []
            for x, y in zip(lane_points, h_samples):
                if x != -2:
                    #  Scale points according to resized image
                    new_x = x * x_scale
                    new_y = y * y_scale
                    points.append([new_x, new_y])
    
            if points:
                if self.format_type == 'parameter':
                    points_np = np.array(points)
                    xs = points_np[:, 0]
                    ys = points_np[:, 1]
    
                    if len(xs) >= 5:
                        ys_min, ys_max = ys.min(), ys.max()
                        ys_norm = (ys - ys_min) / (ys_max - ys_min + 1e-8)
                        xs_norm = xs / new_width
    
                        coeffs = np.polyfit(ys_norm.astype(np.float32), xs_norm.astype(np.float32), deg=4)
    
                        lanes.append({
                            'params': coeffs.tolist(),
                            'offset': float(ys_min),
                            'ys_max': float(ys_max)
                        })
                else:
                    lanes.append({'points': points})   # We only did this- no parameter
    
        if not lanes:
            print(f"[WARNING] No valid lanes found in sample: {sample['raw_file']}")
            return {'lanes': []}
    
        return {'lanes': lanes}

