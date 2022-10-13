from timeit import timeit
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import logging, copy, time
import concurrent.futures
from nilearn.datasets import fetch_abide_pcp, fetch_atlas_harvard_oxford
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib
import matplotlib.pyplot as plt
import h5py

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
        read_abide()
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
"""
def read_abide():
    data_path = Path(_data_base_path).joinpath('ABIDE') # The default data download path = <FedGroup>/data/ABIDE
    Path.mkdir(data_path, exist_ok=True) # Create a new dir if no exist
    
    """ Fetch ABIDE I data and Parcellate it by Harvard-Oxford (HO) atlas
    Input: data_path-> File directory of ABIDE I dataset 
    Return: path_signal_dict-> A dict with {Nifti file path <str>: Transformed signals <np.array>}
            path_site_dict-> Dict with {Nifti file path <str>: Label <int i.e. 1=Austism or 2=Control>}
    """ 
    def _fetch_transform_abide(data_path):

        # fetch_abide_pcp() will download preprocessed ABIDE I data from amazon and return sklearn Bunch with
        # {'description': <str>, description of ABIDE I
        # 'phenotypic': <np.recarray>, additional participants' data
        # 'func_preproc': <list>}, list of scan file proxy paths with nii.gz format (can load by Nibable)
        abide = fetch_abide_pcp(data_dir=data_path, legacy_format=True)
        
        # path_label_dict: {'./Caltech_0051461_func_preproc.nii.gz': 1=Austism or 2=Control, ...}
        path_label_dict = {}
        for idx, fpath in enumerate(abide.func_preproc[-5:]):
            path_label_dict[fpath] = abide.phenotypic['DX_GROUP'][idx]

        # Download Harvard-Oxford atlas
        atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', data_dir=data_path) # It is a 3D deterministic atlas
        # Instantiate the masker with label image and label values
        masker = NiftiLabelsMasker(atlas.maps,
                                labels=atlas.labels,
                                standardize=True)
        
        # Transform the nii.gz scan images to signals
        def _transform_scan2signal(site_path):
            masker_local = copy.deepcopy(masker)
            signal = masker_local.fit_transform(site_path)
            print(Path(site_path).name, ':', signal.shape)
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
        return path_signal_dict, path_label_dict
    
    # Check the exist of cached signal HDF5 file, if there is no cached file in data_path,
    # We need to download ABIDE I dataset and preprocess it.
    cached_filename = 'abide1_transformed_signal.h5'
    cached_h5_path = data_path.joinpath(cached_filename)
    if Path.is_file(cached_h5_path) is False:
        
        # Download ABIDE I data and get transformed signals and labels data
        path_signal_dict, path_label_dict = _fetch_transform_abide(data_path)

        # We can't save the whole path because HDF5 will have confusion of seperator '/',
        # here, we merge data by site name (i.e. UCLA, NYU, ...)
        sitename_data_dict = {}
        for path in path_signal_dict:
            site_name = Path(path).name.split('_')[0]
            # sitename_data_dict-> {sitename: list of tuple like (signal, label, path)}
            signal, label = path_signal_dict[path], path_label_dict[path]
            data_tuple = (signal, label, path)
            sitename_data_dict.setdefault(site_name, []).append(data_tuple)

        # Save these signals data to a HDF5 file
        with h5py.File(cached_h5_path, 'w') as h5f:
            for site, data_list in sitename_data_dict.items():
                # Create a new group with site name
                grp = h5f.require_group(site)
                # Create HDF5 datasets to save signals and labels, note that the timeseries data in x have variable length
                grp.create_dataset('x', data=[data_tuple[0] for data_tuple in data_list])
                grp.create_dataset('y', data=[data_tuple[1] for data_tuple in data_list])
                # We save the time series length for every scan record
                grp.create_dataset('length', data=[data_tuple[0].shape[0] for data_tuple in data_list])
                # We also save the origin path to provide recovery support
                grp.create_dataset('path', data=[data_tuple[2] for data_tuple in data_list])
    """
    with h5py.File(cached_h5_path, 'r') as h5f:
        print(list(h5f.keys()))
        print(h5f['SBL']['x'].shape, h5f['SBL']['y'][:3], h5f['SBL']['length'][:3], h5f['SBL']['path'][:3])
    """

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