# dataio/loader/__init__.py
import json

from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset
from dataio.loader.hms_dataset import HMSDataset
from dataio.loader.cmr_3D_dataset import CMR3DDataset
from dataio.loader.us_dataset import UltraSoundDataset
from dataio.loader.pancreas_small1_dataset import PancreasSmall1Dataset

def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'ukbb_sax': CMR3DDataset,
        'acdc_sax': CMR3DDataset,
        'rvsc_sax': CMR3DDataset,
        'hms_sax': HMSDataset,
        'test_sax': TestDataset,
        'us': UltraSoundDataset,
        'pancreas_small1': PancreasSmall1Dataset
    }[name]

def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """
    return getattr(opts, dataset_name)