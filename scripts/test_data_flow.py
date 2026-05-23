# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

print("Testing data preprocessing flow")
print("=" * 50)

sequence, doy_norm, label = None, None, None

try:
    from data.preprocess import build_time_sequence
    
    data_dir = "./data/landsat_images/2023"
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        sequence, doy_norm = build_time_sequence(data_dir, 2023)
        print("Success: Time sequence built")
        print("  - Shape:", sequence.shape)
        print("  - Time steps:", sequence.shape[0])
        print("  - Features:", sequence.shape[1])
        print("  - Height:", sequence.shape[2])
        print("  - Width:", sequence.shape[3])
        print("  - DOY normalized shape:", doy_norm.shape)
    else:
        print("Warning: Data directory not found or empty:", data_dir)
        print("  Please prepare data first or run scripts/generate_sample_data.py")
    
except Exception as e:
    print("Error: Preprocessing failed:", e)
    import traceback
    traceback.print_exc()

print("")
print("Testing label loading")
print("=" * 50)

try:
    label_path = "./data/labels/crop_label_2023.npy"
    if os.path.exists(label_path):
        label = np.load(label_path)
        print("Success: Label loaded")
        print("  - Shape:", label.shape)
        print("  - Number of classes:", len(np.unique(label)))
        print("  - Class values:", np.unique(label))
    else:
        print("Warning: Label file not found:", label_path)
    
except Exception as e:
    print("Error: Label loading failed:", e)

print("")
print("Testing dataset")
print("=" * 50)

try:
    from data.datasets.crop_dataset import CropDataset
    
    print("Success: Dataset module loaded")
    print("  - Available class: CropDataset")
    
except Exception as e:
    print("Error: Dataset module loading failed:", e)
    import traceback
    traceback.print_exc()

print("")
print("Test completed!")