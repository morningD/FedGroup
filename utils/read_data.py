from termios import VMIN
from timeit import timeit
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import logging, copy, time
import concurrent.futures
import pickle, h5py
from nilearn.datasets import fetch_abide_pcp, fetch_atlas_harvard_oxford
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib
import matplotlib.pyplot as plt

from utils.fedbn_data_utils import DigitsDataset, OfficeDataset, DomainNetDataset

# The default databases path is <FedGroup>/data
# In most cases, no modification is needed
_data_base_path = str(Path(__file__).parent.parent.joinpath('data').absolute())
"""
Load datasets from files
In:     Dataset Name (str)
Out:    List of PyTorch Dataset
"""
def read_federated_data(dsname, data_base_path=None):

    logging.basicConfig(filename='read_data.log', encoding='utf-8', level=logging.DEBUG, \
        format='%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    if data_base_path is not None: _data_base_path = data_base_path
    
    if dsname == 'digits':
        train_loaders, test_loaders = read_digits(percent=1.0, batch=32)
    if dsname == 'office':
        train_loaders, val_loaders, test_loaders = read_office(batch=32)
    if dsname == 'domainnet':
        train_loaders, val_loaders, test_loaders = read_domainnet()
    if dsname == 'abide':
        read_abide(percent=0.5, batch=32)
    pass


""" Below <read_***> functions are borrowed from other FL codes,
    I warp these function and make them return a list of PyTorch Dataloader, one loader to one client,
    You dont need to read these code because they look very inconsistent and hard to maintain,
    unless you need modify their settings.
"""
# Borrow form FedBN
def read_digits(percent=1.0, batch=32):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    data_path = str(Path(_data_base_path).joinpath('MNIST'))
    mnist_trainset     = DigitsDataset(data_path, channels=1, percent=percent, train=True,  transform=transform_mnist)
    # Test labels counter = [1380, 1575, 1398, 1428, 1365, 1263, 1375, 1459, 1365, 1392]
    mnist_testset      = DigitsDataset(data_path, channels=1, percent=percent, train=False, transform=transform_mnist)

    # SVHN
    data_path = str(Path(_data_base_path).joinpath('SVHN'))
    svhn_trainset      = DigitsDataset(data_path, channels=3, percent=percent,  train=True,  transform=transform_svhn)
    # Test labels counter = [1338, 3792, 2947, 2276, 1996, 1853, 1541, 1523, 1341, 1251]
    svhn_testset       = DigitsDataset(data_path, channels=3, percent=percent,  train=False, transform=transform_svhn)

    # USPS
    data_path = str(Path(_data_base_path).joinpath('USPS'))
    usps_trainset      = DigitsDataset(data_path, channels=1, percent=percent,  train=True,  transform=transform_usps)
    # Test labels counter = [311, 254, 186, 165, 170, 143, 167, 158, 142, 164]
    usps_testset       = DigitsDataset(data_path, channels=1, percent=percent,  train=False, transform=transform_usps)

    # Synth Digits
    data_path = str(Path(_data_base_path).joinpath('SynthDigits'))
    synth_trainset     = DigitsDataset(data_path, channels=3, percent=percent,  train=True,  transform=transform_synth)
    # Test labels counter = [9826, 9857, 9706, 9769, 9682, 9798, 9778, 9792, 9805, 9778]
    synth_testset      = DigitsDataset(data_path, channels=3, percent=percent,  train=False, transform=transform_synth)

    # MNIST-M
    data_path = str(Path(_data_base_path).joinpath('MNIST_M'))
    mnistm_trainset     = DigitsDataset(data_path, channels=3, percent=percent,  train=True,  transform=transform_mnistm)
    # Test labels counter = [1380, 1575, 1398, 1428, 1365, 1263, 1375, 1459, 1365, 1392]
    mnistm_testset      = DigitsDataset(data_path, channels=3, percent=percent,  train=False, transform=transform_mnistm)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=batch,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=batch,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=batch,  shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders

# Borrow form FedBN
def read_office(batch=32):
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])
    
    # amazon
    amazon_trainset = OfficeDataset(_data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(_data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(_data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(_data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(_data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(_data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(_data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(_data_base_path, 'webcam', transform=transform_test, train=False)

    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=batch, shuffle=True)
    amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=batch, shuffle=False)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=batch, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=batch, shuffle=True)
    caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=batch, shuffle=False)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=batch, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=batch, shuffle=True)
    dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=batch, shuffle=False)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=batch, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=batch, shuffle=True)
    webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=batch, shuffle=False)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=batch, shuffle=False)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    return train_loaders, val_loaders, test_loaders

# Borrow form FedBN
def read_domainnet():
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])
    
    # clipart
    clipart_trainset = DomainNetDataset(_data_base_path, 'clipart', transform=transform_train)
    clipart_testset = DomainNetDataset(_data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = DomainNetDataset(_data_base_path, 'infograph', transform=transform_train)
    infograph_testset = DomainNetDataset(_data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = DomainNetDataset(_data_base_path, 'painting', transform=transform_train)
    painting_testset = DomainNetDataset(_data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = DomainNetDataset(_data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_testset = DomainNetDataset(_data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = DomainNetDataset(_data_base_path, 'real', transform=transform_train)
    real_testset = DomainNetDataset(_data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = DomainNetDataset(_data_base_path, 'sketch', transform=transform_train)
    sketch_testset = DomainNetDataset(_data_base_path, 'sketch', transform=transform_test, train=False)

    min_data_len = min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset))
    val_len = int(min_data_len * 0.05)
    min_data_len = int(min_data_len * 0.05)

    clipart_valset   = torch.utils.data.Subset(clipart_trainset, list(range(len(clipart_trainset)))[-val_len:])
    clipart_trainset = torch.utils.data.Subset(clipart_trainset, list(range(min_data_len)))
    
    infograph_valset   = torch.utils.data.Subset(infograph_trainset, list(range(len(infograph_trainset)))[-val_len:])
    infograph_trainset = torch.utils.data.Subset(infograph_trainset, list(range(min_data_len)))
    
    painting_valset   = torch.utils.data.Subset(painting_trainset, list(range(len(painting_trainset)))[-val_len:])
    painting_trainset = torch.utils.data.Subset(painting_trainset, list(range(min_data_len)))

    quickdraw_valset   = torch.utils.data.Subset(quickdraw_trainset, list(range(len(quickdraw_trainset)))[-val_len:])
    quickdraw_trainset = torch.utils.data.Subset(quickdraw_trainset, list(range(min_data_len)))

    real_valset   = torch.utils.data.Subset(real_trainset, list(range(len(real_trainset)))[-val_len:])
    real_trainset = torch.utils.data.Subset(real_trainset, list(range(min_data_len)))

    sketch_valset   = torch.utils.data.Subset(sketch_trainset, list(range(len(sketch_trainset)))[-val_len:])
    sketch_trainset = torch.utils.data.Subset(sketch_trainset, list(range(min_data_len)))


    clipart_train_loader = torch.utils.data.DataLoader(clipart_trainset, batch_size=32, shuffle=True)
    clipart_val_loader   = torch.utils.data.DataLoader(clipart_valset, batch_size=32, shuffle=False)
    clipart_test_loader  = torch.utils.data.DataLoader(clipart_testset, batch_size=32, shuffle=False)

    infograph_train_loader = torch.utils.data.DataLoader(infograph_trainset, batch_size=32, shuffle=True)
    infograph_val_loader = torch.utils.data.DataLoader(infograph_valset, batch_size=32, shuffle=False)
    infograph_test_loader = torch.utils.data.DataLoader(infograph_testset, batch_size=32, shuffle=False)

    painting_train_loader = torch.utils.data.DataLoader(painting_trainset, batch_size=32, shuffle=True)
    painting_val_loader = torch.utils.data.DataLoader(painting_valset, batch_size=32, shuffle=False)
    painting_test_loader = torch.utils.data.DataLoader(painting_testset, batch_size=32, shuffle=False)

    quickdraw_train_loader = torch.utils.data.DataLoader(quickdraw_trainset, batch_size=32, shuffle=True)
    quickdraw_val_loader = torch.utils.data.DataLoader(quickdraw_valset, batch_size=32, shuffle=False)
    quickdraw_test_loader = torch.utils.data.DataLoader(quickdraw_testset, batch_size=32, shuffle=False)

    real_train_loader = torch.utils.data.DataLoader(real_trainset, batch_size=32, shuffle=True)
    real_val_loader = torch.utils.data.DataLoader(real_valset, batch_size=32, shuffle=False)
    real_test_loader = torch.utils.data.DataLoader(real_testset, batch_size=32, shuffle=False)

    sketch_train_loader = torch.utils.data.DataLoader(sketch_trainset, batch_size=32, shuffle=True)
    sketch_val_loader = torch.utils.data.DataLoader(sketch_valset, batch_size=32, shuffle=False)
    sketch_test_loader = torch.utils.data.DataLoader(sketch_testset, batch_size=32, shuffle=False)
    

    train_loaders = [clipart_train_loader, infograph_train_loader, painting_train_loader, quickdraw_train_loader, real_train_loader, sketch_train_loader]
    val_loaders = [clipart_val_loader, infograph_val_loader, painting_val_loader, quickdraw_val_loader, real_val_loader, sketch_val_loader]
    test_loaders = [clipart_test_loader, infograph_test_loader, painting_test_loader, quickdraw_test_loader, real_test_loader, sketch_test_loader]

    return train_loaders, val_loaders, test_loaders

""" Autism Brain Imaging Data Exchange I Dataset 
    Dataset URL: https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
    nilearn source code: https://github.com/nilearn/nilearn/blob/d628bf6b/nilearn/datasets/func.py#L857
    Input:  percent-> how many sites' data are involved; batch-> batch_size
"""
def read_abide(percent=0.5, batch=32, strategy='correlation'):
    data_path = Path(_data_base_path).joinpath('ABIDE') # The default data download path = <FedGroup>/data/ABIDE
    Path.mkdir(data_path, exist_ok=True) # Create a new dir if no exist
    
    # Check the exist of cached signal pickle file, if there is no cached file in data_path,
    # we need to download ABIDE I dataset and preprocess it first.
    cached_filename = 'abide1_transformed_signal.pkl'
    cached_pkl_path = data_path.joinpath(cached_filename).absolute()
    
    if Path.is_file(cached_pkl_path) is False:
        # Download ABIDE I data and get transformed signals and labels data
        # sitename_data_dict-> {sitename: list of tuple like (signal, label, length, path)}
        sitename_data_dict = _fetch_transform_abide(cached_pkl_path)
        sitenames = ' '.join(sitename_data_dict.keys())
        logging.debug(f'Download and transform ABIDE data: {sitenames}')

    """ Preprocess the data, including: 
        1, Retain larger sites' data according to 'precent'
        2, Padding the signals to fix length
        3, Train/Test split by the site (split by domain)
        4, Save the preprocessed data
    """
    train_h5_path, test_h5_path =  _preprocess_abide(cached_pkl_path, percent, strategy)

    return cached_pkl_path, train_h5_path, test_h5_path
    

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
    
    # path_label_dict: {'./Caltech_0051461_func_preproc.nii.gz': 1=Austism or 2=Control, ...}
    path_label_dict = {}
    for idx, fpath in enumerate(abide.func_preproc):
        path_label_dict[fpath] = abide.phenotypic['DX_GROUP'][idx]

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
        # sitename_data_dict-> {sitename: list of tuple like (signal, label, length, path)}
        signal, label = path_signal_dict[path], path_label_dict[path]
        data_tuple = (signal, label, signal.shape[0], path)
        sitename_data_dict.setdefault(site_name, []).append(data_tuple)

    # Save these signals data to a pickle file
    with open(cached_pkl_path, 'wb') as pklf:
        pickle.dump(sitename_data_dict, pklf)

    return sitename_data_dict

""" Preprocess ABIDE
"""
def _preprocess_abide(cached_pkl_path, percent, strategy='correlation'):
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
            y = np.array([tuple[1] for tuple in sitename_data_dict[name]], dtype=np.uint8)
            grp.create_dataset('x', data=x)
            grp.create_dataset('y', data=y)
    
    return train_h5_path.absolute(), test_h5_path.absolute()

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

# For debug
def _test_read_digits():
    percent = 1.0
    batch = 32
    train_loaders, test_loader = read_digits(percent, batch)

'''
('i', 'Unnamed: 0', 'SUB_ID', 'X', 'subject', 'SITE_ID', 'FILE_ID', 'DX_GROUP', 'DSM_IV_TR', 'AGE_AT_SCAN', 'SEX', 'HANDEDNESS_CATEGORY', 'HANDEDNESS_SCORES', 'FIQ', 'VIQ', 'PIQ', 'FIQ_TEST_TYPE', 'VIQ_TEST_TYPE', 
'PIQ_TEST_TYPE', 'ADI_R_SOCIAL_TOTAL_A', 'ADI_R_VERBAL_TOTAL_BV', 'ADI_RRB_TOTAL_C', 'ADI_R_ONSET_TOTAL_D', 'ADI_R_RSRCH_RELIABLE', 'ADOS_MODULE', 'ADOS_TOTAL', 'ADOS_COMM', 'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV', 
'ADOS_RSRCH_RELIABLE', 'ADOS_GOTHAM_SOCAFFECT', 'ADOS_GOTHAM_RRB', 'ADOS_GOTHAM_TOTAL', 'ADOS_GOTHAM_SEVERITY', 'SRS_VERSION', 'SRS_RAW_TOTAL', 'SRS_AWARENESS', 'SRS_COGNITION', 'SRS_COMMUNICATION', 'SRS_MOTIVATION', 
'SRS_MANNERISMS', 'SCQ_TOTAL', 'AQ_TOTAL', 'COMORBIDITY', 'CURRENT_MED_STATUS', 'MEDICATION_NAME', 'OFF_STIMULANTS_AT_SCAN', 'VINELAND_RECEPTIVE_V_SCALED', 'VINELAND_EXPRESSIVE_V_SCALED', 'VINELAND_WRITTEN_V_SCALED', 
'VINELAND_COMMUNICATION_STANDARD', 'VINELAND_PERSONAL_V_SCALED', 'VINELAND_DOMESTIC_V_SCALED', 'VINELAND_COMMUNITY_V_SCALED', 'VINELAND_DAILYLVNG_STANDARD', 'VINELAND_INTERPERSONAL_V_SCALED', 'VINELAND_PLAY_V_SCALED', 
'VINELAND_COPING_V_SCALED', 'VINELAND_SOCIAL_STANDARD', 'VINELAND_SUM_SCORES', 'VINELAND_ABC_STANDARD', 'VINELAND_INFORMANT', 'WISC_IV_VCI', 'WISC_IV_PRI', 'WISC_IV_WMI', 'WISC_IV_PSI', 'WISC_IV_SIM_SCALED', 
'WISC_IV_VOCAB_SCALED', 'WISC_IV_INFO_SCALED', 'WISC_IV_BLK_DSN_SCALED', 'WISC_IV_PIC_CON_SCALED', 'WISC_IV_MATRIX_SCALED', 'WISC_IV_DIGIT_SPAN_SCALED', 'WISC_IV_LET_NUM_SCALED', 'WISC_IV_CODING_SCALED', 'WISC_IV_SYM_SCALED', 
'EYE_STATUS_AT_SCAN', 'AGE_AT_MPRAGE', 'BMI', 'anat_cnr', 'anat_efc', 'anat_fber', 'anat_fwhm', 'anat_qi1', 'anat_snr', 'func_efc', 'func_fber', 'func_fwhm', 'func_dvars', 'func_outlier', 'func_quality', 'func_mean_fd', 
'func_num_fd', 'func_perc_fd', 'func_gsr', 'qc_rater_1', 'qc_notes_rater_1', 'qc_anat_rater_2', 'qc_anat_notes_rater_2', 'qc_func_rater_2', 'qc_func_notes_rater_2', 'qc_anat_rater_3', 'qc_anat_notes_rater_3', 
'qc_func_rater_3', 'qc_func_notes_rater_3', 'SUB_IN_SMP')
'''