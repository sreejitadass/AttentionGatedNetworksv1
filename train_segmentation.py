import numpy
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model

from torchvision.datasets import Cityscapes
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

def get_cityscapes_dataset(data_path, split, transform):
    return Cityscapes(
        root=data_path,
        split=split,
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=transform
    )

def get_cityscapes_transforms():
    return Compose([
        Resize((512, 1024)),  # Resize images to a smaller resolution
        ToTensor(),          # Convert to PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained models
    ])


def train(arguments):
    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time2()))
        exit()

    # Setup Data Loader
    # train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData)
    # valid_dataset = ds_class(ds_path, split='val',        transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    # test_dataset  = ds_class(ds_path, split='test',       transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    
    # train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=train_opts.batchSize, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_dataset, num_workers=0, batch_size=train_opts.batchSize, shuffle=False)
    # test_loader  = DataLoader(dataset=test_dataset,  num_workers=0, batch_size=train_opts.batchSize, shuffle=False)

    train_transform = get_cityscapes_transforms()
    valid_transform = get_cityscapes_transforms()

    train_dataset = get_cityscapes_dataset(data_path='data', split='train', transform=train_transform)
    valid_dataset = get_cityscapes_dataset(data_path='data', split='val', transform=valid_transform)
    test_dataset = get_cityscapes_dataset(data_path='data', split='test', transform=valid_transform)

    train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=4, batch_size=train_opts.batchSize, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=train_opts.batchSize, shuffle=False)


    # # Get the first image from the dataset
    first_img = train_loader.dataset[0]

    # image_np = first_img.permute(1, 2, 0).numpy()

    # # Display the image using matplotlib
    # plt.imshow(image_np)
    # plt.axis('off')  # Turn off axis
    # plt.show()

    # return

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                # Visualise predictions
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        # Update the plots
        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()


if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    # parser.add_argument('-c', '--config',  help='training config file', required=True)
    # parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    # args = parser.parse_args()

    class Args:
        config = "configs/config_unet_ct_dsv.json"
        debug = True

    # Create an instance of the Args class
    args = Args()

    train(args)
