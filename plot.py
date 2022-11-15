from cProfile import label
from tkinter import image_names
from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from nilearn.datasets import  fetch_atlas_harvard_oxford
from utils.read_data import read_abide, read_domainnet, read_abide_h5file
import pickle, h5py
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
from itertools import combinations
import concurrent.futures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score
import copy

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

def plot_signals_histogram(pkl_path):
    with open(pkl_path, 'rb') as pf:
        sitename_data_dict = pickle.load(pf)
    print(sitename_data_dict.keys())
    
    sub_sitenames = list(sitename_data_dict.keys())[:5]
    sitename_signal_list = []
    uid = 0
    for site in sub_sitenames:
        for tuple in sitename_data_dict[site]:
            signals_per_uid = tuple[0].reshape(-1).tolist()
            label = tuple[1]
            for s in signals_per_uid:
                sitename_signal_list.append((uid, site, s, label))
            uid += 1

    sites_signals_df = pd.DataFrame(sitename_signal_list, columns=['uid', 'site', 'signal', 'label'])
    fig = plt.figure(figsize=(10,10))
    print(sites_signals_df.groupby('site').var())
    print(sites_signals_df.groupby('site').mean())
    sns.histplot(data=sites_signals_df, x='signal', hue='site', stat= 'probability', element='poly')
    plt.savefig(save_images_path.joinpath(f'hist_ABIDE_signals_all.png'), dpi=500)
    #print(sites_signals_df)

def plot_corr_ROId_heatmap(train_corr_path, atlas, strategy='loo', ext_corr_path=None):
    atlas_image = np.asanyarray(atlas['maps'].dataobj)
    ROI_names = atlas['labels'][1:] # The label:0 is Backgroud
    ROI_labels = list(range(1,len(ROI_names)+1)) # The int label of each ROI
    n_labels = len(ROI_labels)
    affine = atlas['maps'].affine

    # Store the ROIs center coordinates in the reference (anatomy) space
    centers = [] 
    # 1, Calculate the voxel center of each ROI region
    for la in ROI_labels:
        indices = np.argwhere(atlas_image==la)
        c = indices.mean(axis=0)
        centers.append(c)
    centers = np.vstack(centers)

    # 2, Voxel space -> Anatomy spacea
    from nibabel.affines import apply_affine
    anat_centers = apply_affine(affine, centers)
    
    # 3, Calculate the pairwise (anatomy space) distance between two ROI
    from sklearn.metrics.pairwise import euclidean_distances
    distance = euclidean_distances(anat_centers)

    # 4, Load correlations and labels of each site
    n_features = int(n_labels * (n_labels - 1) / 2) # Lower triangular parts of correlation matrix
    target_map = {1:'Austism', 2:'Control'}
    data_dict = read_abide_h5file(train_corr_path, flatten=True, merge=False)
    if ext_corr_path is not None:
        ext_data_dict = read_abide_h5file(ext_corr_path, flatten=True, merge=True)

    # 5, We choose the a control subject's connectome from first site as the typical connection strength
    [corr_vector, target_vector] = list(data_dict.values())[-1]
    for idx, label in enumerate(target_vector.tolist()):
        if label == 2:
            typical_conn = corr_vector[idx, :] # shape = (n_features,)
            break

    ''' IMPORTANT Configurationsn '''
    # We group features to bins, the setting is:
    min_d, max_d, num_dbins = 0, 150, 20 # Distance bins
    min_c, max_c, num_cbins = -0.5, +1.0, 20 # Connectome bin
    d_iterval, c_iterval = (max_d - min_d)/num_dbins, (max_c - min_c)/num_cbins 
    
    ''' # DEBUG CODE
    [ext_corr_vector, ext_target_vector] = data_dict['UM']
    data_dict = {i:data_dict[i] for i in data_dict if i in ['Leuven']}
    '''
    # We calculate the accuarcy group by bin for each site
    for site_name, [corr_vector, target_vector] in data_dict.items():
        bin_dict = {}
        # Bin each feature, for each feature, we calcualte the bin number,
        # note: idx, x, y start from 0 (the first ROI region, no blackground)
        for idx, x, y in zip(range(n_features), *np.tril_indices(n_labels, k=-1)):
            conn_strength = typical_conn[idx]
            cbin = 0 if conn_strength < 0 else min(int((conn_strength-min_c)/c_iterval), num_cbins-1)
            conn_distance = distance[x][y]
            dbin = 0 if conn_distance < 0 else min(int((conn_distance-min_d)/d_iterval), num_dbins-1)
            bin_dict.setdefault((cbin, dbin), []).append(idx)

        rst_list = []
        for c in range(num_cbins):
            for d in range(num_dbins):
                if (c,d) in bin_dict:
                    fids = bin_dict[(c,d)]
                else:
                    fids = []
                    bin_dict.setdefault((c, d), [])
                
                # Train a Linear classifier for each site and each features (connectome)
                if strategy == 'loo':
                    # Using Leave-One-Out strategy
                    rst = _LOO_fit_test(corr_vector[:, fids], target_vector)
                    print(f'Site: {site_name}, BIN-{c}-{d}, Acc: {rst}')
                elif strategy == 'external':
                    # Use the data from external site for testing
                    [ext_corr_vector, ext_target_vector] = next(iter(ext_data_dict.values()))
                    rst = _external_fit_test(corr_vector[:, fids], target_vector, ext_corr_vector[:, fids], ext_target_vector)
                    print(f'Site: {site_name}, BIN-{c}-{d}, F1-score: {rst}')
                else:
                    pass
                rst_list.append((c, d, rst, fids, len(fids)))
                
        df = pd.DataFrame(rst_list, columns=['cbin', 'dbin', 'result', 'feature_ids', 'num_features'])
        # Save results
        df.to_csv(save_images_path.joinpath(f'corr-ROId-{strategy}_{site_name}.csv'))

        # We plot the heatmap of each site
        df = pd.read_csv(save_images_path.joinpath(f'corr-ROId-{strategy}_{site_name}.csv'))
        fig = plt.figure(figsize=(10,9))
        ax = sns.heatmap(df.pivot('cbin', 'dbin', 'result'), cmap='coolwarm', \
            vmin=0.40, vmax=0.70, center=0.5, linewidths=1, linecolor='black')
        ax.invert_yaxis()
        plt.savefig(save_images_path.joinpath(f'corr-ROId-{strategy}_{site_name}.png'), dpi=500)


    ''' # Plot Atlas image
    fig = plt.figure()
    plt.imshow(atlas_image[26,:,:].T, cmap='gray', origin='lower')
    plt.savefig(save_images_path.joinpath(f'test_scan.png'), dpi=500)
    quit()
    '''

'''# Deprecated: Multi-thread can't work with sklearn, I dont know the reason
# Warp the fit_test function with cbin, dbin info
def _calculate_bin_wrapper(argv):
    # Just need cbin, dbin information from each thread
    cbin, dbin = argv[0], argv[1]
    fit_test_arg = argv[2:]
    print(f'Handling {cbin}-{dbin}')
    return cbin, dbin, _fit_test(*fit_test_arg)
'''

# Define the Leave-One-Out train and test procedure
def _LOO_fit_test(data, label):
    if data is None or data.size == 0:
        acc = np.nan # Return NaN if no features in this bin
        return acc
    
    # Leave-One-Out ensemblly train linear regression model
    y_pred = []
    n_samples, n_features = data.shape[0], data.shape[1]
    #estimator = LinearRegression()
    estimator = KNeighborsClassifier(3)
    
    for train_idx, test_idx, in LeaveOneOut().split(data):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        # We just use this feature to predict the label if there only has one feature
        if n_features == 1:
            # Note: X_train only has one feature
            reg = estimator.fit(X_train, y_train).predict(X_test)
            # target_map = {1:'Austism', 2:'Control'}
            pred = 1 if abs(reg-1) < abs(reg-2) else 2
        
        # We use any two features to predict the label, the average regression score is used to prediction
        if n_features > 1:
            ensem_preds = []
            for fids in combinations(range(n_features), 2):
                reg = estimator.fit(X_train[:, fids], y_train).predict(X_test[:, fids])
                # target_map = {1:'Austism', 2:'Control'}
                ensem_preds.append(1 if abs(reg-1) < abs(reg-2) else 2)
            pred = 1 if len([i for i in ensem_preds if i == 1]) > len(ensem_preds)/2 else 2
        y_pred.append(pred)
        
    #acc = accuracy_score(label, y_pred) # Accuracy is ill-suited for comparison of imbalanced dataset
    score = f1_score(label, y_pred)
    return score

def _external_fit_test(data, label, ext_data, ext_label):

    if data is None or data.size == 0:
        f1 = np.nan # Return NaN if no features in this bin
        return f1
    
    n_samples, n_features = data.shape[0], data.shape[1]
    # Ensemblly train linear regression model
    #estimator = LinearRegression()
    # KNN model
    estimator = KNeighborsClassifier(n_neighbors=5)
    
    X_train, X_test = data, ext_data
    y_train, y_test = label, ext_label

    # We just use this feature to predict the label if there only has one feature
    if n_features == 1:
        # Note: X_train only has one feature, X_test.shape = (n_samples, 1)
        regs = estimator.fit(X_train, y_train).predict(X_test)
        # target_map = {1:'Austism', 2:'Control'}
        preds = np.array([1 if abs(reg-1) < abs(reg-2) else 2 for reg in regs])
    
    # We use any two features to predict the label, the average regression score is used to prediction
    if n_features > 1:
        preds_list = []
        for fids in combinations(range(n_features), 2):
            regs = estimator.fit(X_train[:, fids], y_train).predict(X_test[:, fids])
            # target_map = {1:'Austism', 2:'Control'}
            preds_list.append(np.array([1 if abs(reg-1) < abs(reg-2) else 2 for reg in regs]))

        preds = np.zeros(y_test.size)
        for idx in range(y_test.size):
            preds[idx] = 1 if len([1 for subpreds in preds_list if subpreds[idx] == 1]) > len(preds_list)/2 else 2
    
    f1 = f1_score(y_test, preds, pos_label=1) # Note: pos_label is Austism=1
    return f1

#plot_correlation_matrix(train_h5_path, atlas)
#plot_correlation_matrix(test_h5_path, atlas)
#plot_ABIDE_raw_tsne(train_h5_path)
#plot_domainnet_raw_tsne()
#plot_svhn_raw_tsne()
#plot_mnist_raw_tsne()
#plot_signals_histogram(cached_pkl_path)
#plot_corr_ROId_heatmap(train_h5_path, atlas, strategy='external', ext_corr_path=test_h5_path)
plot_corr_ROId_heatmap(train_h5_path, atlas, strategy='loo', ext_corr_path=test_h5_path)
"""
    import seaborn as sns
    plt.figure(figsize=(12,10))
    plot = sns.heatmap(corr,vmin=-1,vmax=1, cmap='coolwarm')
    plt.savefig('{}_correlation.png'.format(name), dpi=400)
"""