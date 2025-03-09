# dataio/loader/utils.py
import numpy as np
import nibabel as nib
import os

def load_nifti_img(filepath, dtype=np.float32):
    """
    Load a NIfTI image file.

    Args:
        filepath (str): Path to the NIfTI file.
        dtype (type): Data type for the output array (default: np.float32).

    Returns:
        tuple: (image_data, affine_matrix)
    """
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_fdata(), dtype=dtype)  # Use get_fdata() instead of get_data()
    affine = nim.affine
    return out_nii_array, affine

def write_nifti_img(data, meta, out_dir):
    """
    Write a 3D volume as a NIfTI file.

    Args:
        data (np.ndarray): The 3D volume data.
        meta (dict): Metadata including 'affine', 'pixdim', 'dim', 'name'.
        out_dir (str): Output directory.
    """
    affine = meta['affine']  # Use the affine directly (already a NumPy array)
    pixdim = meta['pixdim']
    dim = meta['dim']
    filename = meta['name']
    filepath = os.path.join(out_dir, filename)

    # Create NIfTI image
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms(pixdim[:3])  # Use only the first 3 values for zooms (x, y, z)
    nib.save(img, filepath)

    return filepath

def check_exceptions(volume, label):
    """
    Check for exceptions in volume and label data.
    """
    if volume is None or label is None:
        raise ValueError("Volume or label data is None")
    if volume.shape != label.shape:
        raise ValueError(f"Shape mismatch: volume {volume.shape} vs label {label.shape}")

def is_image_file(filename):
    """
    Check if a file is an image file.
    """
    return filename.lower().endswith(('.nii', '.nii.gz', '.dcm'))