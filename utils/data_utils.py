import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import concurrent.futures
import copy
import logging
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from nilearn.datasets import fetch_abide_pcp, fetch_atlas_harvard_oxford
from nilearn.maskers import NiftiLabelsMasker
from PIL import Image
from torch.utils.data import Dataset


class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

        # We can collect and print the TRAIN/TEST labels class distribution here
        log_labels_counter(train, self.labels)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('{}/office_caltech_10/{}_train.pkl'.format(base_path, site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('{}/office_caltech_10/{}_test.pkl'.format(base_path, site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else './data'

        # We can collect and print the TRAIN/TEST labels class distribution here
        log_labels_counter(train, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('{}/DomainNet/{}_train.pkl'.format(base_path, site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('{}/DomainNet/{}_test.pkl'.format(base_path, site), allow_pickle=True)
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}
                
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else './data'

        # We can collect and print the TRAIN/TEST labels class distribution here
        log_labels_counter(train, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class ABIDE1Dataset(Dataset):
    def __init__(self, sitename, data, label, train=True, transform=None):
            
        self.label_dict = {'autism':1, 'control':0}
        self.transform = transform
        self.train = train
        self.sitename = sitename
        self.images = data.astype(np.float32)
        self.labels = label.astype(np.uint8) # 0 ->austism, 1->control

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

''' Read the abide data dict from h5py file, the format is consistent with _preprocess_abide() '''
def read_abide_h5file(h5_path, flatten=False, merge=False):
    data_dict = {}
    with h5py.File(h5_path, 'r') as h5f:
        for site_name, grp in h5f.items():
            data_X = np.array(grp['x'])
            data_y = np.array(grp['y']) # 1=Austism or 2=Control
            
            if flatten == True: 
                # Flatten the X to (n_samples, n_features)
                data_X = data_X.reshape(data_X.shape[0], -1)
            data_dict[site_name] = [data_X, data_y]
    
    if merge == True:
        merged_sitenames = '-'.join(data_dict.keys())
        data_X_list, data_y_list = [], []
        for [data_X, data_y] in data_dict.values():
            data_X_list.append(data_X)
            data_y_list.append(data_y)
        print(merged_sitenames, np.unique(np.hstack(data_y_list), return_counts=True))
        return dict({merged_sitenames: [np.vstack(data_X_list), np.hstack(data_y_list)]})

    return data_dict

# We make this function to create datasets by once file read
def create_abide_datasets(h5_path, transform, is_train):
    data_dict = read_abide_h5file(h5_path, flatten=False, merge=False)
    datasets = []
    for site, [data_X, data_y] in data_dict.items():
        datasets.append(ABIDE1Dataset(site, data_X, data_y, is_train, transform))
    return datasets


""" Fetch ABIDE I data and Parcellate it by Harvard-Oxford (HO) atlas
    Input: data_path-> File directory of ABIDE I dataset 
    Return: path_signal_dict-> A dict with {Nifti file path <str>: Transformed signals <np.array>}
            path_site_dict-> Dict with {Nifti file path <str>: Label <int i.e. 1=Austism or 2=Control>}
""" 
def _fetch_transform_abide(cached_pkl_path):
    # Save directory of ABIDE I data
    data_path = cached_pkl_path.parent
    # fetch_abide_pcp() will download preprocessed ABIDE I data from amazon and return sklearn Bunch with
    # {'description': <str>, description of ABIDE I
    # 'phenotypic': <np.recarray>, additional participants' data
    # 'func_preproc': <list>}, list of scan file proxy paths with nii.gz format (can load by Nibable)
    abide = fetch_abide_pcp(data_dir=data_path, legacy_format=True)
    
    # path_phenotypic_dict: {'./Caltech_0051461_func_preproc.nii.gz': phenotypic {'DX_GROUP':1=Austism or 2=Control, 'SEX': 1=Male or 2=Female}, ...}
    path_phenotypic_dict = {}
    for idx, fpath in enumerate(abide.func_preproc):
        path_phenotypic_dict[fpath] = abide.phenotypic[:][idx]

    # Download Harvard-Oxford atlas
    atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', data_dir=data_path) # It is a 3D deterministic atlas
    # Instantiate the masker with label image and label values
    masker = NiftiLabelsMasker(atlas.maps,
                            labels=atlas.labels,
                            standardize=True)
    
    # Transform the nii.gz scan images to signals
    def _transform_scan2signal(site_path, verbose=True):
        masker_local = copy.deepcopy(masker)
        signal = masker_local.fit_transform(site_path)
        if verbose:
            print('Transformed:', Path(site_path).name, ':', signal.shape) # Debug
        del masker_local
        return site_path, signal
    
    path_signal_dict = {}
    # This multithread implementation take 74.9s for 20 files with 4 workers
    # We use ThreadPoolExecutor to parral the parallelize the image to signal operation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:        
        for result in executor.map(_transform_scan2signal, abide.func_preproc):
            path, signal = result[0], result[1]
            path_signal_dict[path] = signal

    """ This single thread implementation take 112.8s for 20 files
    for path in abide.func_preproc:
        signal = masker.fit_transform(path)
        path_signal_dict[path] = signal
    """

    # Here, we merge data by site name (i.e. UCLA, NYU, ...)
    sitename_data_dict = {}
    for path in path_signal_dict:
        site_name = Path(path).name.split('_')[0]
        # sitename_data_dict-> {sitename: list of tuple like (signal, phenotypic, length, path)}
        signal, phenotypic = path_signal_dict[path], path_phenotypic_dict[path]
        data_tuple = (signal, phenotypic, signal.shape[0], path)
        sitename_data_dict.setdefault(site_name, []).append(data_tuple)

    # Save these signals data to a pickle file
    with open(cached_pkl_path, 'wb') as pklf:
        pickle.dump(sitename_data_dict, pklf)

    return sitename_data_dict

""" Preprocess ABIDE
    label should be listed in ABIDE I datalegend: http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf
"""
def _preprocess_abide(cached_pkl_path, percent, strategy='correlation', label='DX_GROUP'):

    def preprocess_phenotypic(phenotypic, label):
        if label == "DX_GROUP": # From 1=Austism, 2=Control to 1=Austism (Positive), 0=Control
            return abs(phenotypic[label] - 2)
        if label == 'SEX': # From 1=Male, 2=Female to 0=Male, 2=Female
            return phenotypic[label] - 1
        if label == 'AGE_AT_SCAN':
            age_min, age_max = 6.47, 64
            return (phenotypic[label] - age_min) / (age_max-age_min)

    with open(cached_pkl_path, 'rb') as pklf:
        sitename_data_dict = pickle.load(pklf)
    
    # Sort the data dict with scan file size large->small
    sitename_data_dict = dict(sorted(sitename_data_dict.items(), key=lambda item: len(item[1]), reverse=True))
    keep_site_count = int(len(sitename_data_dict) * percent)
    # The names of site will be retained large->small
    keep_site_name = list(sitename_data_dict.keys())[:keep_site_count]
    sitenames = ' '.join(keep_site_name)
    logging.debug(f'Retained site names: {sitenames}')

    # Pad the signals to same length
    if strategy == 'timeseries':    
        # The max number of scan images (time series length)
        max_len = 300
        logging.debug(f'The max number of scan images is {max_len}')

        for name in keep_site_name:
            for idx, data_tuple in enumerate(sitename_data_dict[name]):
                data_len = data_tuple[2]
                
                if data_len > max_len:
                    # Truncate the time series to (max_len, num_features)
                    truncated_signal = data_tuple[0][:max_len]
                    sitename_data_dict[name][idx] = (truncated_signal, *data_tuple[1:])
                    logging.debug('The scan data in {} has been truncated to {:d}'.format(
                        sitename_data_dict[name][idx][3], max_len))
                else:
                    # Paded the series to max length, new shape = (max_len, num_features)
                    pad_width = int(max_len - data_len)
                    # Pad with zero value according to:
                    # https://github.com/nilearn/nilearn/blob/98a3ee060b55aa073118885d11cc6a1cecd95059/nilearn/regions/signal_extraction.py#L128
                    pad_signal = np.pad(data_tuple[0], ((pad_width, 0), (0, 0)), 'constant', constant_values=0)
                    sitename_data_dict[name][idx] = (pad_signal, *data_tuple[1:])
                    logging.debug('The scan data in {} has been padded to {:d}'.format(
                        sitename_data_dict[name][idx][3], max_len))
    
    if strategy == 'correlation':
        # Correlation[-1, +1] -> [-∞, +∞], Ref: https://en.wikipedia.org/wiki/Fisher_transformation
        # Use API from nilearn: <nilearn.connectome.ConnectivityMeasure>, LedoitWolf estimator is used
        for name in keep_site_name:
            from nilearn.connectome import ConnectivityMeasure

            # Only use lower triangular parts of correlation matrix and w/o diagonal elements, flatten into 1D
            # i.e. The correlation value in (2,1) (3,1) (3,2) (4,1) (4,2) (4,3) ...
            connv_measure = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
            connectivity = connv_measure.fit_transform([tuple[0] for tuple in sitename_data_dict[name]])
            # Repalce the signals data with functional connectivity
            for idx, data_tuple in enumerate(sitename_data_dict[name]):
                # add newaxis to reshape the 1D correlation into 2D
                sitename_data_dict[name][idx] = (connectivity[idx][np.newaxis,:], *data_tuple[1:])
            
            """ Deprecated: The maximum-liklihood estimate will yield some zero variaces, so we use LedoitWolf instead.
            fisher_transormation = True
            for idx, data_tuple in enumerate(sitename_data_dict[name]):
                # We just use maximum-liklihood estimate
                corr = np.corrcoef(data_tuple[0], rowvar=False)
                coor_uptri = corr[np.triu_indices(corr.shape[0], k=1)].reshape(1,-1)
                connectivity = np.arctanh(coor_uptri) if fisher_transormation else coor_uptri
            """

    # Train/Test split
    test_ratio = 1/3
    train_h5_path = Path(cached_pkl_path).parent.joinpath(f'abide1_{strategy}_train.h5')
    test_h5_path = Path(cached_pkl_path).parent.joinpath(f'abide1_{strategy}_test.h5')
    K = int(keep_site_count * (1- test_ratio))
    # Largest K sites for training, remains for testing
    with h5py.File(train_h5_path, 'w') as ta, h5py.File(test_h5_path, 'w') as te:
        for name in keep_site_name:
            h5writer = ta if name in keep_site_name[:K] else te
            grp = h5writer.require_group(name)
            x = np.stack([tuple[0] for tuple in sitename_data_dict[name]], axis=0)
            y = np.array([preprocess_phenotypic(tuple[1], label) for tuple in sitename_data_dict[name]], dtype=np.uint8)
            grp.create_dataset('x', data=x)
            grp.create_dataset('y', data=y)
    
    return train_h5_path.absolute(), test_h5_path.absolute()


# We can collect and print the labels class distribution by this function
def log_labels_counter(train, labels):
    prefix = 'TRAIN' if train else 'TEST'
    logging.debug(prefix+' CLS DIST:'+str(np.unique(labels, return_counts=True)))