import random
import os
import json
from datasets.lane_dataset import TuSimpleDataset
from utils.tokenizer import LaneTokenizer
from utils.visualizer import draw_lanes
from PIL import Image
import cv2
import torch
import yaml
import numpy as np
from collections import Counter

# Load config
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Special tokens mapping for readability
SPECIAL_TOKENS = {
    0: "<START>",
    1: "<END>",
    2: "<FORMAT_SEGMENTATION>",
    3: "<FORMAT_ANCHOR>",
    4: "<FORMAT_PARAMETER>",
    5: "<LANE_SEPARATOR>"
}

def interpret_tokens(sequence):
    readable = []
    for token in sequence:
        if token in SPECIAL_TOKENS:
            readable.append(SPECIAL_TOKENS[token])
        else:
            readable.append(str(token))
    return readable

# Denormalize the image tensor for visualization
def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)

def test_tokenizer(dataset_path, format_type='anchor', nbins=1000, num_samples=5):
    # Init tokenizer
    tokenizer = LaneTokenizer(nbins=nbins)

    # Load dataset (augmentation is inside already)
    dataset = TuSimpleDataset(
        root_dir=dataset_path,
        split='train',
        nbins=nbins,
        format_type=format_type,
        config=config
    )

    # Output folder
    output_dir = f"test_outputs_{format_type}"
    os.makedirs(output_dir, exist_ok=True)

    # Pick random samples
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        # Fetch sample (augmented image and sequences)
        sample = dataset[idx]

        # Extract components
        image = sample['image']  # Tensor: (3, H, W)
        image_np = (denormalize(image).permute(1, 2, 0).numpy() * 255).astype('uint8')
        input_seq = sample['input_seq'].tolist()
        target_seq = sample['target_seq'].tolist()
        raw_file = sample['raw_file']
        '''
        # === DEBUG ===
        print(f"\n=== [DEBUG] Sample index: {idx} ===")
        print(f"[DEBUG] Raw file: {raw_file}")
        print(f"[DEBUG] Input sequence length: {len(input_seq)}")
        print(f"[DEBUG] Target sequence length: {len(target_seq)}")
        print(f"[DEBUG] Input token distribution: {Counter(input_seq)}")
        print(f"[DEBUG] Target token distribution: {Counter(target_seq)}")
        print(f"[DEBUG] Image min: {image_np.min()}, max: {image_np.max()}")
        '''
        # Decode sequence back to annotation (what model learned, augmented)
        decoded_annotation = tokenizer.decode(
            input_seq,
            (dataset.image_size[1], dataset.image_size[0]),
            format_type=format_type
        )

        # === Prepare original image (before augmentation) ===
        original_sample = dataset.samples[idx]
        original_image_path = os.path.join(dataset.root_dir, original_sample['raw_file'])
        original_image = Image.open(original_image_path).convert('RGB')
        original_size = original_image.size  # Original size (width, height)

        # Resize original image to target size
        original_image_resized = original_image.resize((dataset.image_size[1], dataset.image_size[0]), Image.BILINEAR)
        original_image_np = np.array(original_image_resized)
        
        # Prepare original annotation (pre-augmentation)
        original_annotation = dataset._convert_annotation(
            original_sample,
            original_size=original_size,
            target_size=(dataset.image_size[1], dataset.image_size[0])
        )
        
        #original_annotation = sample['annotation']

        # === Save sequence JSON ===
        json_output = {
            "raw_file": raw_file,
            "input_sequence_tokens": input_seq,
            "input_sequence_readable": interpret_tokens(input_seq),
            "target_sequence_tokens": target_seq,
            "target_sequence_readable": interpret_tokens(target_seq),
            "decoded_annotation": decoded_annotation,
            "original_annotation": original_annotation
        }
        sequence_json_path = os.path.join(output_dir, f"sample_{idx}_{format_type}_sequence.json")
        with open(sequence_json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved sequence JSON to {sequence_json_path}")

        # === Visualization ===

        # Draw original lanes (green) on original image
        original_image_vis = draw_lanes(original_image_np.copy(), original_annotation['lanes'], color=(0, 255, 0))

        # Draw decoded lanes (red) on augmented image
        decoded_image_vis = draw_lanes(image_np.copy(), decoded_annotation, color=(0, 0, 255))

        # Side-by-side comparison
        concatenated = Image.fromarray(cv2.hconcat([original_image_vis, decoded_image_vis]))
        side_by_side_output = os.path.join(output_dir, f"sample_{idx}_{format_type}_sidebyside.jpg")
        concatenated.save(side_by_side_output)
        print(f"Saved side-by-side comparison to {side_by_side_output}")

        # Overlay visualization: overlay original annotation (green) and decoded (red) on augmented image
        overlay_image = draw_lanes(image_np.copy(), original_annotation['lanes'], color=(0, 255, 0))
        overlay_image = draw_lanes(overlay_image, decoded_annotation, color=(0, 0, 255))
        overlay_output = os.path.join(output_dir, f"sample_{idx}_{format_type}_overlay.jpg")
        Image.fromarray(overlay_image).save(overlay_output)
        print(f"Saved overlay comparison to {overlay_output}")

    print(f"\n✅ All samples processed! Output folder: {output_dir}")

if __name__ == "__main__":
    dataset_path = "../archive/TUSimple/train_set"
    format_type = "segmentation"  # Change to "segmentation", "anchor", or "parameter"
    test_tokenizer(dataset_path, format_type=format_type)
