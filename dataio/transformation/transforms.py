import torchvision.transforms as T
import torch
import numpy as np
from pprint import pprint


class Transformations:

    def __init__(self, name):
        self.name = name

        # Input patch and scale size
        self.scale_size = (192, 192, 1)
        self.patch_size = (128, 128, 1)

        # Affine and Intensity Transformations
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.random_flip_prob = 0.0

        # Divisibility factor for testing
        self.division_factor = (16, 16, 1)

    def initialise(self, opts):
        """
        Update transformation parameters based on options.
        """
        t_opts = getattr(opts, self.name, None)
        if not t_opts:
            return

        # Update all parameters if they exist in the options
        if hasattr(t_opts, 'scale_size'): self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'patch_size'): self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'): self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate_val'): self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'): self.scale_val = t_opts.scale
        if hasattr(t_opts, 'inten_val'): self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'division_factor'): self.division_factor = t_opts.division_factor

    def get_transformation(self):
        return {
            'ukbb_sax': self.cmr_3d_sax_transform,
            'hms_sax':  self.hms_sax_transform,
            'test_sax': self.test_3d_sax_transform,
            'acdc_sax': self.cmr_3d_sax_transform,
            'us':       self.ultrasound_transform,
        }[self.name]()

    def pad_to_size(self, size):
        def pad(image):
            pad_h = max(0, size[0] - image.shape[-2])
            pad_w = max(0, size[1] - image.shape[-1])
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            return T.functional.pad(image, padding)
        return pad
    
    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def crop_to_size(self, size):
        def crop(image):
            _, h, w = image.shape
            start_h = (h - size[0]) // 2
            start_w = (w - size[1]) // 2
            return image[:, start_h:start_h + size[0], start_w:start_w + size[1]]
        return crop

    def normalise_percentile(self, norm_flag=(True, False)):
        def normalise(image):
            if norm_flag[0]:
                percentile_99 = np.percentile(image.numpy(), 99)
                image = torch.clamp(image / percentile_99, 0, 1)
            return image
        return normalise

    def cmr_3d_sax_transform(self):
        train_transform = T.Compose([
            self.pad_to_size(self.scale_size[:2]),
            T.ToTensor(),
            T.RandomHorizontalFlip(p=self.random_flip_prob),
            T.RandomVerticalFlip(p=self.random_flip_prob),
            T.RandomAffine(
                degrees=self.rotate_val,
                translate=self.shift_val,
                scale=self.scale_val,
                interpolation=T.InterpolationMode.BILINEAR
            ),
            self.normalise_percentile(),
            T.Lambda(lambda x: x.unsqueeze(0)),  # Add channel
            T.RandomCrop(self.patch_size[:2]),
        ])

        valid_transform = T.Compose([
            self.pad_to_size(self.scale_size[:2]),
            T.ToTensor(),
            self.normalise_percentile(),
            self.crop_to_size(self.patch_size[:2]),
        ])

        return {'train': train_transform, 'valid': valid_transform}

    def ultrasound_transform(self):
        train_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.unsqueeze(0)),  # Add channel
            self.crop_to_size(self.patch_size[:2]),
            T.RandomHorizontalFlip(p=self.random_flip_prob),
            T.RandomAffine(
                degrees=self.rotate_val,
                translate=self.shift_val,
                scale=self.scale_val,
                interpolation=T.InterpolationMode.BILINEAR
            ),
            T.Normalize(mean=[0.5], std=[0.5]),  # Example normalization
        ])

        valid_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.unsqueeze(0)),  # Add channel
            self.crop_to_size(self.patch_size[:2]),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        return {'train': train_transform, 'valid': valid_transform}

    def hms_sax_transform(self):

        # Training transformation
        # 2D Stack input - 3D High Resolution output segmentation

        train_transform = []
        valid_transform = []

        # First pad to a fixed size
        # Torch tensor
        # Channels first
        # Joint affine transformation
        # In-plane respiratory motion artefacts (translation and rotation)
        # Random Crop
        # Normalise the intensity range
        train_transform = T.Compose([])

        return {'train': train_transform, 'valid': valid_transform}


    def test_3d_sax_transform(self):
        test_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.unsqueeze(0)),  # Add channel
            self.normalise_percentile(),
        ])

        return {'test': test_transform}