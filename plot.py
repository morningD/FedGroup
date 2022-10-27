from cProfile import label
from tkinter import image_names
from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from nilearn.datasets import  fetch_atlas_harvard_oxford
from utils.read_data import read_abide, read_domainnet
import pickle, h5py
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler

# Read ABIDE data
cached_pkl_path, train_h5_path, test_h5_path = read_abide(percent=0.5, strategy='correlation')
# Configurations of this plot script
save_images_path = Path(__file__).parent.joinpath('plot_imgs').absolute()
data_base_path = Path(cached_pkl_path).parent.parent.absolute()
abide_data_path = Path(cached_pkl_path).parent.absolute()
atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', data_dir=abide_data_path)

def plot_signals():
    pass

def plot_correlation_matrix(corr_path, altas):
    altas_labels = altas['labels'][1:] # index0 is Background
    n_labels = len(altas_labels)
    n_features = int(n_labels * (n_labels - 1) / 2) # Lower triangular parts of correlation matrix
    target_map = {1:'Austism', 2:'Control'}
    with h5py.File(corr_path, 'r') as h5f:
        for site_name, grp in h5f.items():
            corr_vector = np.array(grp['x'])
            target_vector = np.array(grp['y']) # 1=Austism or 2=Control
            n_samples = corr_vector.shape[0]
            corr_vector = corr_vector.reshape(n_samples, n_features)
            corr_matrix = np.zeros((n_samples, n_labels, n_labels))
            for idx in range(n_samples):
                corr_matrix[idx][np.tril_indices(n_labels, k=-1)] = corr_vector[idx]
            corr_mean = corr_matrix.mean(axis=0)
            fig, axes = plt.subplots(1,2, figsize=(24,10))
            ax1, ax2 = axes.flatten()
            img = plotting.plot_matrix(corr_matrix[0], labels=altas_labels, colorbar=False, tri='lower', axes=ax1,\
                vmax=1.0, vmin=-1.0, title=f'Correlation, <{target_map[target_vector[0]]}>')
            plt.colorbar(img, ax=ax1)
            img = plotting.plot_matrix(corr_mean, labels=altas_labels, colorbar=False, tri='lower', axes=ax2,\
                vmax=1.0, vmin=-1.0, title=f'Correlation Mean')
            plt.colorbar(img, ax=ax2)
            plt.savefig(save_images_path.joinpath(f'correlation_{site_name}.png'), dpi=500)
            
def plot_ABIDE_raw_tsne(corr_path):
    target_map = {1:'Austism', 2:'Control'}
    all_x, all_y, all_sites = [], [], []
    with h5py.File(corr_path, 'r') as h5f:
        for site_name, grp in h5f.items():
            corr_x = np.array(grp['x'])
            target_y = np.array(grp['y']) # 1=Austism or 2=Control
            n_samples = corr_x.shape[0]
            corr_x = corr_x.reshape(n_samples, -1) # reshape to 2D

            all_x.append(corr_x)
            all_y.append(target_y)
            all_sites.append(np.array([len(all_sites)+1]*n_samples))
            #all_sites.append(target_vector)

            corr_x = StandardScaler().fit_transform(corr_x)
        
            embs = TSNE(n_components=2, learning_rate='auto',init='pca', perplexity=3,\
                random_state=114514).fit_transform(corr_x)
            fig = plt.figure(figsize=(5,5))
            sns.scatterplot(x=embs[:,0], y=embs[:,1], hue=target_y)
            plt.savefig(save_images_path.joinpath(f'tsne_ABIDE_raw_{site_name}.png'), dpi=500)

    all_x = np.vstack(all_x)
    all_x = StandardScaler().fit_transform(all_x)
    all_y = np.hstack(all_y)
    all_sites = np.hstack(all_sites)
    embs = TSNE(n_components=2, learning_rate='auto',init='pca', perplexity=5,\
        random_state=114514, n_jobs=16).fit_transform(all_x)

    fig = plt.figure(figsize=(5,5))
    sns.scatterplot(x=embs[:,0], y=embs[:,1], hue=all_sites, palette=sns.color_palette("hls", all_sites[-1]))
    #sns.scatterplot(x=embs[:,0], y=embs[:,1], hue=all_y, palette=sns.color_palette("hls", 2))
    plt.savefig(save_images_path.joinpath(f'tsne_ABIDE_raw_all.png'), dpi=500)

def plot_dense_embs_tsne(train_path, test_path):
    pass

def plot_domainnet_raw_tsne():
    train_loaders, val_loaders, test_loaders = read_domainnet()
    n_domains = len(train_loaders)
    all_x, all_y, all_domains = [], [], []
    for idx, domain_loader in enumerate(train_loaders):
        for batch_x, batch_y in domain_loader:
            npbx = batch_x.numpy()
            batch_size = npbx.shape[0]
            npbx = npbx.reshape(batch_size, -1)
            npbx = StandardScaler().fit_transform(npbx)
            all_x.append(npbx)
            all_y.append(batch_y.numpy().reshape(-1,))
            all_domains.append([idx]*batch_size)
            #all_domains.append(batch_y.numpy().reshape(-1,))
    all_x = np.vstack(all_x)
    #all_x = StandardScaler().fit_transform(all_x)
    all_y = np.hstack(all_y)
    all_domains = np.hstack(all_domains)
    embs = TSNE(n_components=2, learning_rate='auto',init='pca', perplexity=10,\
        random_state=114514, n_jobs=16).fit_transform(all_x)

    fig = plt.figure(figsize=(10,10))
    #sns.scatterplot(x=embs[:,0], y=embs[:,1], hue=all_domains, palette=sns.color_palette("hls", n_domains))
    sns.scatterplot(x=embs[:,0], y=embs[:,1], hue=all_y, palette=sns.color_palette("hls", np.unique(all_y).shape[0]))
    plt.savefig(save_images_path.joinpath(f'tsne_domainnet_raw_all.png'), dpi=500)

def plot_svhn_raw_tsne():
    from torchvision.datasets import SVHN
    SVHN_dataset = SVHN(data_base_path, split='test', download=True)
    data, labels = SVHN_dataset.data.astype(np.float32), SVHN_dataset.labels
    n_samples = data.shape[0]
    n_classes = np.unique(labels).shape[0]
    # Greyscale
    #data = data.reshape(n_samples,3,-1)
    #data[:,0] = 0.2989*data[:,0] + 0.5870*data[:,1] + 0.1140*data[:,2] 
    #data = data[:,0]
    
    data = data.reshape(n_samples, -1)

    # Standardlization
    data = StandardScaler().fit_transform(data / 255.0)

    # Truncated SVD
    data = TruncatedSVD(50, random_state=114514).fit_transform(data)
    
    # t-SNE embedding
    embs = TSNE(n_components=2, learning_rate='auto',init='pca', perplexity=30,\
        random_state=114514, n_jobs=16).fit_transform(data)

    fig = plt.figure(figsize=(10,10))
    sns.scatterplot(x=embs[:,0], y=embs[:,1], hue=labels, s=9, palette=sns.color_palette("hls", n_classes))
    plt.savefig(save_images_path.joinpath(f'tsne_SVHN_raw_all.png'), dpi=500)

def plot_mnist_raw_tsne():
    from torchvision.datasets import MNIST
    MNIST_dataset = MNIST(data_base_path, download=True)
    data, targets = MNIST_dataset.data.numpy().astype(np.float32), MNIST_dataset.targets.numpy()
    n_samples = data.shape[0]
    n_classes = np.unique(targets).shape[0]
    data = data.reshape(n_samples, -1)
    #data = StandardScaler().fit_transform(data / 255.0)
    embs = TSNE(n_components=2, learning_rate='auto',init='pca', perplexity=20,\
        random_state=114514, n_jobs=32).fit_transform(data)

    fig = plt.figure(figsize=(10,10))
    sns.scatterplot(x=embs[:,0], y=embs[:,1], hue=targets, palette=sns.color_palette("hls", n_classes))
    plt.savefig(save_images_path.joinpath(f'tsne_MNIST_raw_all.png'), dpi=500)

#plot_correlation_matrix(train_h5_path, atlas)
#plot_correlation_matrix(test_h5_path, atlas)
#plot_ABIDE_raw_tsne(train_h5_path)
#plot_domainnet_raw_tsne()
plot_svhn_raw_tsne()
#plot_mnist_raw_tsne()
"""
    import seaborn as sns
    plt.figure(figsize=(12,10))
    plot = sns.heatmap(corr,vmin=-1,vmax=1, cmap='coolwarm')
    plt.savefig('{}_correlation.png'.format(name), dpi=400)
"""