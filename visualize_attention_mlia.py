from torch.utils.data import DataLoader

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from models import get_model

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math, numpy, os
from skimage.transform import resize
from dataio.loader.utils import write_nifti_img
from torch.nn import functional as F
import numpy as np

def visualize_attention(args):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, f"configs/{args.config}.json")
    checkpoint_name = args.checkpoint
    experiment_name = args.experiment
    checkpoint_load_path = os.path.join(current_dir, f"checkpoints/{experiment_name}/{checkpoint_name}.pth")


    # Load options
    json_opts = json_file_to_pyobj(config_path)

    # Setup the NN Model
    model = get_model(json_opts.model)
    model.load_network_from_path(model.net, checkpoint_load_path, strict=True)

    # Setup Dataset and Augmentation
    dataset_class = get_dataset(json_opts.training.arch_type)
    dataset_path = get_dataset_path(json_opts.training.arch_type, json_opts.data_path)
    dataset_transform = get_dataset_transformation(json_opts.training.arch_type, json_opts.augmentation)

    # Setup Data Loader
    dataset = dataset_class(dataset_path, split='test', transform=dataset_transform['valid'], preload_data=True)
    data_loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    # test
    for iteration, (input_arr, _) in enumerate(data_loader, 1):
        model.set_input(input_arr)
        if iteration != 5:
            continue
        layer_names = ['attentionblock2', 'attentionblock3', 'attentionblock4']
        for layer_name in layer_names:
            inp_fmap, out_fmap = model.get_feature_maps(layer_name=layer_name, upscale=False)

            # Display the input image and Down_sample the input image
            orig_input_img = model.input.permute(2, 3, 4, 1, 0).cpu().numpy()
            upsampled_attention   = F.upsample(out_fmap[0], size=input_arr.size()[2:], mode='trilinear').data.squeeze().permute(1,2,3,0).cpu().numpy()[0,:,:,0]
            upsampled_fmap_before = F.upsample(inp_fmap[0], size=input_arr.size()[2:], mode='trilinear').data.squeeze().permute(1,2,3,0).cpu().numpy()[0,:,:,0]
            upsampled_fmap_after  = F.upsample(out_fmap[1], size=input_arr.size()[2:], mode='trilinear').data.squeeze().permute(1,2,3,0).cpu().numpy()[0,:,:,0]
        
            # Define the directories
            save_directory = f'visualization/feature_maps/{json_opts.model.experiment_name}/{checkpoint_name}/{layer_name}/'
            if not os.path.isdir(save_directory):
                os.makedirs(save_directory)

            input_img = input_arr[0, 0, 0, :, :]
            normalized_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
            normalized_upsampled_attention = (upsampled_attention - upsampled_attention.min()) / (upsampled_attention.max() - upsampled_attention.min())
            normalized_upsampled_fmap_before = (upsampled_fmap_before - upsampled_fmap_before.min()) / (upsampled_fmap_before.max() - upsampled_fmap_before.min())
            normalized_upsampled_fmap_after = (upsampled_fmap_after - upsampled_fmap_after.min()) / (upsampled_fmap_after.max() - upsampled_fmap_after.min())
            

            plt.imshow(normalized_img, cmap='gray', interpolation='none')  # Grayscale base
            plt.imshow(normalized_upsampled_attention, cmap='jet', alpha=0.25, interpolation='none') 
            plt.tight_layout() 
            plt.savefig(f'{save_directory}upsampled_attention_{iteration}.png')
            plt.clf()

            plt.imshow(normalized_img, cmap='gray', interpolation='none')  # Grayscale base
            plt.imshow(normalized_upsampled_fmap_before, cmap='jet', alpha=0.25, interpolation='none')  
            plt.tight_layout()
            plt.savefig(f'{save_directory}upsampled_fmap_before_{iteration}.png')
            plt.clf()

            plt.imshow(normalized_img, cmap='gray', interpolation='none')  # Grayscale base
            plt.imshow(normalized_upsampled_fmap_after, cmap='jet', alpha=0.25, interpolation='none')  
            plt.tight_layout()
            plt.savefig(f'{save_directory}upsampled_fmap_after_{iteration}.png')
            plt.clf()
        break


    model.destructor()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Attention Visualizer')

    parser.add_argument('-c', '--config',  help='config file name', required=True)
    parser.add_argument('-p', '--checkpoint',   help='checkpoint name', required=True)
    parser.add_argument('-e', '--experiment',   help='experiment name', required=True)
    args = parser.parse_args()

    visualize_attention(args)

