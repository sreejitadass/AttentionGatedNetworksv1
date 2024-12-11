import collections
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.error_logger import ErrorLogger
from models import get_model

def tune_segmentation():
    study = optuna.create_study()
    study.optimize(objective, n_trials=15)
    return study.best_params, study.best_value

def objective(trial):
    # Parse input arguments
    n_epochs = 3
    batchSize = trial.suggest_categorical('batchSize', [8, 16])
    feature_scale = trial.suggest_categorical('feature_scale', [2, 4, 8])
    l2_reg_weight = trial.suggest_loguniform('l2_reg_weight', 1e-7, 1e-5)
    lr_rate = trial.suggest_loguniform('lr_rate', 1e-5, 1e-3)
    shift = trial.suggest_uniform('shift', 0.0, 0.2)
    rotate = trial.suggest_uniform('rotate', 0.0, 30.0)
    random_flip_prob = trial.suggest_uniform('random_flip_prob', 0.0, 0.5)
    optim = trial.suggest_categorical('optim', ['adam', 'sgd'])
    
    # Load options
    json_opts_dict = json.load(open(args.config))

    # set params
    json_opts_dict["training"]["n_epochs"] = n_epochs
    json_opts_dict["training"]["batchSize"] = batchSize
    json_opts_dict["model"]["feature_scale"] = feature_scale
    json_opts_dict["model"]["l2_reg_weight"] = l2_reg_weight
    json_opts_dict["model"]["lr_rate"] = lr_rate
    json_opts_dict["augmentation"]["segmentation"]["shift"] = [shift, shift]
    json_opts_dict["augmentation"]["segmentation"]["rotate"] = rotate
    json_opts_dict["augmentation"]["segmentation"]["random_flip_prob"] = random_flip_prob
    json_opts_dict["model"]["optim"] = optim

    json_opts = json_dict_to_pyobj(json_opts_dict)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model)

    # Setup Data Loader
    train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData)
    valid_dataset = ds_class(ds_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    test_dataset  = ds_class(ds_path, split='test',       transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

    error_logger = ErrorLogger()
    score = 0

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
                metrics = model.get_segmentation_stats()
                error_logger.update({**errors, **metrics}, split=split)

        score = error_logger.get_errors('validation')['Mean_IOU']

        error_logger.reset()

        # Update the model learning rate
        model.update_learning_rate()
    
    return score


def json_dict_to_pyobj(dic):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(json.dumps(dic))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    args = parser.parse_args()

    best_params, best_scores = tune_segmentation()
    print('Best Parameters:', best_params)
    print('Best Score:', best_scores)
