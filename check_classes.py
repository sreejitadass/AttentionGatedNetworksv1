import nibabel as nib
import numpy as np

label = nib.load('/fab3/btech/2022/sreejita.das22b/AttentionGatedNetworksv1/Pancreas_Small1/labels/label0001.nii.gz').get_fdata()
print("Unique values in label:", np.unique(label))