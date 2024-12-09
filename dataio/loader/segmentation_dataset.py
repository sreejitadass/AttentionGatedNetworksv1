# TODO: 
import torch.utils.data as data
import torch
import numpy as np
import datetime

from os import listdir
from os.path import join
from .utils import check_exceptions, load_mhd_image, is_mhd_file

class SegmentationDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(SegmentationDataset, self).__init__()

        if split == 'test':
            split_dir = 'Testing'
        else:
            split_dir = 'Training'

        image_dir = join(root_dir, split_dir, 'Brains')
        target_dir = join(root_dir, split_dir, 'Labels')
        self.image_filenames  = sorted([join(image_dir, x) for x in listdir(image_dir) if is_mhd_file(x)])
        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_mhd_file(x)])
        assert len(self.image_filenames) == len(self.target_filenames)

        split_point = int(.9*len(self.target_filenames))
        if split == 'train':
            self.image_filenames = self.image_filenames[:split_point]
            self.target_filenames = self.target_filenames[:split_point]
        elif split == 'validation':
            self.image_filenames = self.image_filenames[split_point:]
            self.target_filenames = self.target_filenames[split_point:]

        # report the number of images in the dataset
        print('Number of {0} images: {1} MHDs'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [load_mhd_image(ii)[0] for ii in self.image_filenames]
            self.raw_labels = [load_mhd_image(ii)[0] for ii in self.target_filenames]
            print('Loading is done\n')

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input, _, _ = load_mhd_image(self.image_filenames[index])
            target, _, _ = load_mhd_image(self.target_filenames[index])
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])


        # handle exceptions
        check_exceptions(input, target)
        if self.transform:
            input, target = self.transform(input, target)
            input = torch.stack([input] * 8, dim=1) #hacky solution
            target = torch.stack([target] * 8, dim=1) #hacky solution
            print(f'Input shape: {input.shape}')
            print(f'Target Dimension: {target.shape}')

        return input, target

    def __len__(self):
        return len(self.image_filenames)

if __name__ == '__main__':
    dataset = SegmentationDataset("Segmentation_data",'train')

    from torch.utils.data import DataLoader, sampler
    ds = DataLoader(dataset=dataset, num_workers=1, batch_size=2)