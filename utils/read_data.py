import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
from pathlib import Path
from utils.fedbn_data_utils import DigitsDataset, OfficeDataset, DomainNetDataset
import logging
from nilearn.datasets import fetch_abide_pcp


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
        fetch_abide()
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
def fetch_abide():
    data_path = Path(_data_base_path).joinpath('ABIDE') # The default data download path = <FedGroup>/data/ABIDE
    Path.mkdir(data_path, exist_ok=True) # Create a new dir if no exist
    # fetch_abide_pcp will return sklearn Bunch with 
    # {'description': <str>, 'phenotypic': <np.recarray>, func_preproc: <list>}
    data = fetch_abide_pcp(data_dir=data_path, legacy_format=True)
    for k in data:
        print(k, type(data[k]))
    print(data['func_preproc'][0])
    print(data['phenotypic'].dtype.names)
    print(np.unique(data['phenotypic']['SITE_ID'], return_counts=True))
    
    pass
    

# For debug
def _test_read_digits():
    percent = 1.0
    batch = 32
    train_loaders, test_loader = read_digits(percent, batch)