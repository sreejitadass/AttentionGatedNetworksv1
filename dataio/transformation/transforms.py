import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from pprint import pprint

class Transformations:
    def __init__(self, name):
        self.name = name
        self.scale_size = (160, 160, 96)
        self.patch_size = (160, 160, 96)
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.inten_val = (1.0, 1.0)
        self.random_flip_prob = 0.5
        self.division_factor = (16, 16, 1)

    def get_transformation(self):
        return {
            'ukbb_sax': self.cmr_3d_sax_transform,
            'hms_sax': self.hms_sax_transform,
            'test_sax': self.test_3d_sax_transform,
            'acdc_sax': self.cmr_3d_sax_transform,
            'us': self.ultrasound_transform,
            'pancreas_small1': self.pancreas_small1_transform
        }[self.name]()

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts):
        t_opts = getattr(opts, self.name)
        if hasattr(t_opts, 'scale_size'): self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'patch_size'): self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'): self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate_val'): self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'): self.scale_val = t_opts.scale
        if hasattr(t_opts, 'inten_val'): self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'division_factor'): self.division_factor = t_opts.division_factor

    # Custom Transforms
    class ToTensor:
        def __call__(self, data):
            volume, label = data
            volume = torch.from_numpy(volume).float()
            label = torch.from_numpy(label).long()
            return volume, label

    class ChannelsFirst:
        def __call__(self, data):
            volume, label = data
            if volume.dim() == 3:
                volume = volume.unsqueeze(0)
            if label.dim() == 3:
                label = label.unsqueeze(0)
            return volume, label

    class Resize3D:
        def __init__(self, size):
            self.size = size

        def __call__(self, data):
            volume, label = data
            volume = volume.unsqueeze(0)
            label = label.unsqueeze(0)
            volume = F.interpolate(volume, size=self.size, mode='trilinear', align_corners=False)
            label = F.interpolate(label.float(), size=self.size, mode='nearest')
            return volume.squeeze(0), label.squeeze(0).long()

    class Pad3D:
        def __init__(self, size):
            self.size = size

        def __call__(self, data):
            volume, label = data
            h, w, d = volume.shape[-3], volume.shape[-2], volume.shape[-1]
            target_h, target_w, target_d = self.size

            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            pad_d = max(0, target_d - d)

            volume = F.pad(volume, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant', value=0)
            label = F.pad(label, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant', value=0)
            return volume, label

    class RandomFlip3D:
        def __init__(self, h=True, v=True, d=True, p=0.5):
            self.h = h
            self.v = v
            self.d = d
            self.p = p

        def __call__(self, data):
            volume, label = data
            if self.h and np.random.random() < self.p:
                volume = torch.flip(volume, dims=[-2])
                label = torch.flip(label, dims=[-2])
            if self.v and np.random.random() < self.p:
                volume = torch.flip(volume, dims=[-3])
                label = torch.flip(label, dims=[-3])
            if self.d and np.random.random() < self.p:
                volume = torch.flip(volume, dims=[-1])
                label = torch.flip(label, dims=[-1])
            return volume, label

    class RandomAffine3D:
        def __init__(self, rotation_range, translation_range, zoom_range, interp=('bilinear', 'nearest')):
            self.rotation_range = rotation_range
            self.translation_range = translation_range
            self.zoom_range = zoom_range
            self.interp = interp  # Updated to use 'bilinear' for volume, 'nearest' for label

        def __call__(self, data):
            volume, label = data
            batch_size = 1  # Since we're processing one sample at a time

            # Generate random transformations
            angle_z = np.random.uniform(-self.rotation_range, self.rotation_range) * np.pi / 180  # Rotation around z-axis (H-W plane)
            translation = [np.random.uniform(-t, t) for t in self.translation_range] + [0]  # (shift_h, shift_w, shift_d)
            scale = np.random.uniform(self.zoom_range[0], self.zoom_range[1])

            # Create the 3x3 rotation matrix for the z-axis (H-W plane)
            cos_z = np.cos(angle_z)
            sin_z = np.sin(angle_z)
            rotation_matrix = torch.tensor([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z,  0],
                [0,     0,      1]
            ], dtype=torch.float32)

            # Scaling matrix
            scale_matrix = torch.tensor([
                [scale, 0,     0],
                [0,     scale, 0],
                [0,     0,     1]
            ], dtype=torch.float32)

            # Combine rotation and scaling into a 3x3 matrix
            transform_matrix = torch.matmul(rotation_matrix, scale_matrix)

            # Extend to a 3x4 affine matrix by adding the translation
            translation = torch.tensor(translation, dtype=torch.float32)  # (shift_h, shift_w, shift_d)
            transform_matrix = torch.cat([transform_matrix, translation.unsqueeze(1)], dim=1)  # Shape: (3, 4)

            # Add batch dimension for F.affine_grid: (3, 4) -> (1, 3, 4)
            transform_matrix = transform_matrix.unsqueeze(0)

            # Create the affine grid
            grid = F.affine_grid(
                transform_matrix,
                size=(batch_size, volume.shape[0], volume.shape[1], volume.shape[2], volume.shape[3]),
                align_corners=False
            ).to(volume.device)

            # Apply the transformation with appropriate modes
            volume = F.grid_sample(volume.unsqueeze(0), grid, mode=self.interp[0], padding_mode='zeros', align_corners=False).squeeze(0)
            label = F.grid_sample(label.unsqueeze(0).float(), grid, mode=self.interp[1], padding_mode='zeros', align_corners=False).squeeze(0).long()

            return volume, label

    class NormalizeMedic:
        def __init__(self, norm_flag=(True, False)):
            self.norm_flag = norm_flag

        def __call__(self, data):
            volume, label = data
            if self.norm_flag[0]:
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            return volume, label

    class RandomCrop3D:
        def __init__(self, size):
            self.size = size

        def __call__(self, data):
            volume, label = data
            h, w, d = volume.shape[-3:]
            th, tw, td = self.size

            if h < th or w < tw or d < td:
                raise ValueError(f"Volume size {volume.shape} smaller than crop size {self.size}")

            x = np.random.randint(0, h - th + 1)
            y = np.random.randint(0, w - tw + 1)
            z = np.random.randint(0, d - td + 1)

            volume = volume[:, x:x+th, y:y+tw, z:z+td]
            label = label[:, x:x+th, y:y+tw, z:z+td]
            return volume, label

    class SpecialCrop3D:
        def __init__(self, size, crop_type=0):
            self.size = size
            self.crop_type = crop_type

        def __call__(self, data):
            volume, label = data
            h, w, d = volume.shape[-3:]
            th, tw, td = self.size

            if h < th or w < tw or d < td:
                raise ValueError(f"Volume size {volume.shape} smaller than crop size {self.size}")

            if self.crop_type == 0:
                x = (h - th) // 2
                y = (w - tw) // 2
                z = (d - td) // 2
            else:
                x, y, z = 0, 0, 0

            volume = volume[:, x:x+th, y:y+tw, z:z+td]
            label = label[:, x:x+th, y:y+tw, z:z+td]
            return volume, label

    def pancreas_small1_transform(self):
        train_transform = transforms.Compose([
            self.ToTensor(),
            self.ChannelsFirst(),
            self.Resize3D(size=self.scale_size),
            self.Pad3D(size=self.scale_size),
            self.RandomFlip3D(h=True, v=True, d=True, p=self.random_flip_prob),
            self.RandomAffine3D(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
            self.NormalizeMedic(norm_flag=(True, False)),
            self.RandomCrop3D(size=self.patch_size),
        ])

        valid_transform = transforms.Compose([
            self.ToTensor(),
            self.ChannelsFirst(),
            self.Resize3D(size=self.scale_size),
            self.Pad3D(size=self.scale_size),
            self.NormalizeMedic(norm_flag=(True, False)),
            self.SpecialCrop3D(size=self.patch_size, crop_type=0),
        ])

        return {'train': train_transform, 'valid': valid_transform}

    def cmr_3d_sax_transform(self):
        train_transform = transforms.Compose([
            self.ToTensor(),
            self.ChannelsFirst(),
            self.Pad3D(size=self.scale_size),
            self.RandomFlip3D(h=True, v=True, p=self.random_flip_prob),
            self.RandomAffine3D(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
            self.NormalizeMedic(norm_flag=(True, False)),
            self.RandomCrop3D(size=self.patch_size),
        ])

        valid_transform = transforms.Compose([
            self.ToTensor(),
            self.ChannelsFirst(),
            self.Pad3D(size=self.scale_size),
            self.NormalizeMedic(norm_flag=(True, False)),
            self.SpecialCrop3D(size=self.patch_size, crop_type=0),
        ])

        return {'train': train_transform, 'valid': valid_transform}

    def hms_sax_transform(self):
        train_transform = transforms.Compose([])
        valid_transform = transforms.Compose([])
        return {'train': train_transform, 'valid': valid_transform}

    def test_3d_sax_transform(self):
        test_transform = transforms.Compose([
            self.ToTensor(),
            self.ChannelsFirst(),
            self.Pad3D(size=self.scale_size),
            self.NormalizeMedic(norm_flag=(True, False)),
        ])
        return {'test': test_transform}

    def ultrasound_transform(self):
        train_transform = transforms.Compose([
            self.ToTensor(),
            self.ChannelsFirst(),
            self.RandomFlip3D(h=True, v=False, p=self.random_flip_prob),
            self.RandomAffine3D(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
            self.NormalizeMedic(norm_flag=(True, False)),
        ])

        valid_transform = transforms.Compose([
            self.ToTensor(),
            self.ChannelsFirst(),
            self.NormalizeMedic(norm_flag=(True, False)),
        ])

        return {'train': train_transform, 'valid': valid_transform}