import os
import sys
import time
import traceback

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))
from CancerDetection.scripts.infer_restructure import CancerSegmentation
from params import poll_interval_seconds
from utils.utils import get_models_dict, get_models_paths, save_new_patients, list_all_patients
from time import time as ctime
import numpy as np
import torch

def get_voxel_volume(patient):
    subject = patient.subject
    spacing = subject.spacing
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    return voxel_volume

def get_avg_tumor_volume(patient, voxel_volume):
    subject = patient.subject
    segs_tumor_volume = {}
    total_tumor_volume = 0
    for label in list(subject.keys()):
        if 'segmentation' in label:
            seg = subject[label]
            tumor_volume = torch.sum(seg.data[seg.data > 0]) * voxel_volume
            print(f"Tumor volume for patient {patient.short_id} is {tumor_volume:.2f} mm^3")
            segs_tumor_volume[label] = tumor_volume
            total_tumor_volume += tumor_volume
    avg_tumor_volume = total_tumor_volume / len(segs_tumor_volume)
    return avg_tumor_volume, segs_tumor_volume

def main():
    patients = list_all_patients()
    # instead get ground truth data from training examples, load them
    # using torchio and then calculate the volume, other stats then
    # get back to fix segmentation problem
    for patient in patients:
        if patient.contains_seg():
            patient.download_and_preprocess(include_seg=False)
            voxel_volume = get_voxel_volume(patient)
            avg_tumor_volume, segs_tumor_volume = get_avg_tumor_volume(patient, voxel_volume)
            print(f"Average Tumor volume for patient {patient.short_id}: {avg_tumor_volume:.2f} mm^3")

if __name__ == "__main__":
    main()






