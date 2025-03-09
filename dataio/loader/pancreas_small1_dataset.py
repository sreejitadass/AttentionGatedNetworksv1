import os
import glob
import numpy as np
import nibabel as nib
import pydicom
import argparse
from torch.utils.data import Dataset

# Adjust import based on whether running as a script or module
try:
    from .utils import check_exceptions
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataio.loader.utils import check_exceptions

class PancreasSmall1Dataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None, preload_data=False):
        """
        Initialize the Pancreas Small 1 Dataset.

        Args:
            base_dir (str): Base directory containing 'PANCREAS_XXXX' directories and 'labels' subdirectory.
            split (str): Dataset split ('train', 'validation', or 'test').
            transform: Transformation to apply to the data.
            preload_data (bool): Whether to preload all data into memory.
        """
        assert split in ['train', 'validation', 'test'], f"Invalid split: {split}"
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.preload_data = preload_data

        # Define paths
        self.image_dir = base_dir  # Directly use base_dir since images are at the root level
        self.label_dir = os.path.join(base_dir, 'labels')

        # Load label files (flat directory with .nii.gz files)
        self.label_files = sorted(glob.glob(os.path.join(self.label_dir, 'label*.nii.gz')))

        # Check if label directory has files
        if not self.label_files:
            raise ValueError(f"No label files found in {self.label_dir} with pattern 'label*.nii.gz'")

        # Debugging: Print label files
        print(f"Label files found: {self.label_files}")

        # Load image directories (nested structure: PANCREAS_XXXX/UID1/UID2/*.dcm)
        # Find all PANCREAS_XXXX directories
        pancreas_dirs = sorted(glob.glob(os.path.join(self.image_dir, 'PANCREAS_*')))
        self.image_dirs = []

        # Debugging: Print PANCREAS directories
        print(f"PANCREAS directories found: {pancreas_dirs}")

        # Pair image directories with label files based on XXXX identifier
        label_basenames = [os.path.basename(f).replace('label', '').replace('.nii.gz', '') for f in self.label_files]
        print(f"Label basenames (XXXX): {label_basenames}")

        for pancreas_dir in pancreas_dirs:
            pancreas_id = os.path.basename(pancreas_dir).replace('PANCREAS_', '')
            print(f"Checking PANCREAS directory: {pancreas_dir}, ID: {pancreas_id}")
            # Check if this directory contains DICOM files
            dicom_files = sorted(glob.glob(os.path.join(pancreas_dir, '**', '*.dcm'), recursive=True))
            if not dicom_files:
                print(f"Warning: No DICOM files found in {pancreas_dir}, skipping...")
                continue
            if pancreas_id in label_basenames:
                print(f"Match found: PANCREAS_{pancreas_id} with label{pancreas_id}.nii.gz")
                self.image_dirs.append(pancreas_dir)
            else:
                print(f"No matching label found for PANCREAS_{pancreas_id}")

        # Check if any image directories were found
        if not self.image_dirs:
            raise ValueError(f"No matching PANCREAS_XXXX directories with DICOM files found in {self.image_dir} that correspond to label files")

        # Ensure the number of image directories matches the number of label files
        if len(self.image_dirs) != len(self.label_files):
            print(f"Warning: Mismatch between number of image directories ({len(self.image_dirs)}) and label files ({len(self.label_files)})")
            # Keep only the pairs that match
            paired_image_dirs = []
            paired_label_files = []
            label_basenames = [os.path.basename(f).replace('label', '').replace('.nii.gz', '') for f in self.label_files]
            for pancreas_dir in self.image_dirs:
                pancreas_id = os.path.basename(pancreas_dir).replace('PANCREAS_', '')
                if pancreas_id in label_basenames:
                    label_idx = label_basenames.index(pancreas_id)
                    paired_image_dirs.append(pancreas_dir)
                    paired_label_files.append(self.label_files[label_idx])
            self.image_dirs = paired_image_dirs
            self.label_files = paired_label_files

        assert len(self.image_dirs) == len(self.label_files), \
            f"Mismatch between number of image directories ({len(self.image_dirs)}) and label files ({len(self.label_files)})"

        # Split the dataset
        total_files = len(self.image_dirs)
        train_split = 0.7
        val_split = 0.15
        # Compute indices
        train_end = int(total_files * train_split)  # e.g., 7 for 10 files
        val_end = train_end + int(total_files * val_split)  # e.g., 7 + 1 = 8

        # Ensure at least 1 sample per split if total_files > 0
        if total_files > 0:
            train_end = max(1, train_end)
            val_end = max(train_end + 1, val_end)
            val_end = min(val_end, total_files)

        print(f"Total files: {total_files}, train_end: {train_end}, val_end: {val_end}")

        if split == 'train':
            self.image_dirs = self.image_dirs[:train_end]
            self.label_files = self.label_files[:train_end]
            print(f"Train split - Image dirs: {self.image_dirs}")
        elif split == 'validation':
            self.image_dirs = self.image_dirs[train_end:val_end]
            self.label_files = self.label_files[train_end:val_end]
            print(f"Validation split - Image dirs: {self.image_dirs}")
        else:  # test
            self.image_dirs = self.image_dirs[val_end:]
            self.label_files = self.label_files[val_end:]
            print(f"Test split - Image dirs: {self.image_dirs}")

        assert len(self.image_dirs) == len(self.label_files), \
            f"Mismatch after splitting: number of image directories ({len(self.image_dirs)}) and label files ({len(self.label_files)})"
        if len(self.image_dirs) == 0:
            raise ValueError(f"No files available for split '{split}' after splitting (total files: {total_files})")

        # Preload data if specified
        self.data = []
        if self.preload_data:
            for img_dir, lbl_path in zip(self.image_dirs, self.label_files):
                volume = self._load_dicom_volume(img_dir)
                label = nib.load(lbl_path).get_fdata()
                self.data.append((volume, label))

    def _load_dicom_volume(self, dicom_dir):
        """
        Load a 3D volume from a directory containing DICOM files.

        Args:
            dicom_dir (str): Path to the directory containing DICOM files (e.g., PANCREAS_XXXX/UID1/UID2/*.dcm).

        Returns:
            np.ndarray: 3D volume as a numpy array.
        """
        # Find all .dcm files in the nested directory structure
        dicom_files = sorted(glob.glob(os.path.join(dicom_dir, '**', '*.dcm'), recursive=True))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")

        # Read DICOM files and sort by InstanceNumber or slice position
        slices = []
        for dcm_file in dicom_files:
            dcm = pydicom.dcmread(dcm_file)
            slices.append(dcm)

        # Sort slices by InstanceNumber (or SliceLocation if InstanceNumber is unavailable)
        slices.sort(key=lambda x: x.get('InstanceNumber', x.get('SliceLocation', 0)))

        # Stack the slices into a 3D volume
        try:
            volume = np.stack([dcm.pixel_array for dcm in slices], axis=-1)
        except Exception as e:
            raise RuntimeError(f"Error stacking DICOM slices from {dicom_dir}: {e}")

        return volume.astype(np.float32)

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.image_dirs)

    def __getitem__(self, index):
        """
        Fetch a sample from the dataset.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: (volume, label) pair, potentially transformed.

        Raises:
            RuntimeError: If an error occurs while fetching the data.
        """
        try:
            if self.preload_data:
                volume, label = self.data[index]
            else:
                img_dir = self.image_dirs[index]
                lbl_path = self.label_files[index]
                volume = self._load_dicom_volume(img_dir)
                label = nib.load(lbl_path).get_fdata()

            # Handle shape mismatch by cropping to the minimum depth
            if volume.shape != label.shape:
                min_depth = min(volume.shape[2], label.shape[2])
                volume = volume[:, :, :min_depth]
                label = label[:, :, :min_depth]
                print(f"Warning: Cropped volume and label to depth {min_depth} for index {index} "
                      f"(original shapes: volume {volume.shape}, label {label.shape})")

            check_exceptions(volume, label)

            if self.transform:
                volume, label = self.transform((volume, label))

            return volume, label

        except Exception as e:
            raise RuntimeError(f"Error fetching item {index} from dataset: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Pancreas Small 1 Dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'],
                        help='Dataset split to test (default: test)')
    parser.add_argument('--base_dir', type=str, default='/fab3/btech/2022/sreejita.das22b/AttentionGatedNetworksv1/Pancreas_Small1',
                        help='Base directory of the dataset (default: /fab3/.../Pancreas_Small1)')
    args = parser.parse_args()

    dataset = PancreasSmall1Dataset(
        base_dir=args.base_dir,
        split=args.split,
        transform=None,
        preload_data=False
    )
    print(f"Number of samples: {len(dataset)}")
    volume, label = dataset[0]
    print(f"Sample shapes: volume {volume.shape}, label {volume.shape}")