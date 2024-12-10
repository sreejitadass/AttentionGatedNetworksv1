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

def plotNNFilter(units, figure_id, interp='bilinear', colormap=cm.jet, colormap_lim=None):
    plt.ion()
    filters = units.shape[2]
    n_columns = round(math.sqrt(filters))
    n_rows = math.ceil(filters / n_columns) + 1
    fig = plt.figure(figure_id, figsize=(n_rows*3,n_columns*3))
    fig.clf()

    for i in range(filters):
        ax1 = plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(units[:,:,i].T, interpolation=interp, cmap=colormap)
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.colorbar()
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

# Load options
json_opts = json_file_to_pyobj('/home/bhx5gh/Documents/MLIA/MLIA-Attention-Gated-Networks/configs/config_unet_segmentation.json')

# Setup the NN Model

checkpoint_name = '005_net_S'
model = get_model(json_opts.model)
model.load_network_from_path(model.net, 
f'/home/bhx5gh/Documents/MLIA/MLIA-Attention-Gated-Networks/checkpoints/experiment_unet_grid_gating_segmentation/{checkpoint_name}.pth',
strict=True)

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
#if iteration == 1: break