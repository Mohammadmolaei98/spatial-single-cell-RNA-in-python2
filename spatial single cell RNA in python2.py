import sys
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse import isspparse

# Set plotting style
plt.style.use('seaborn-whitegrid')
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
sc.set_figure_params(facecolor="white", figsize=(8, 6))

# Set paths to data
sp_data_folder = '/home/student/Documents/python-project/mouse_brain_visium_wo_cloupe_data/rawdata/ST8059048/'
count_file = '/home/student/Documents/python-project/mouse_brain_visium_wo_cloupe_data/rawdata/ST8059048/filtered_feature_bc_matrix.h5'

# 1. Data Loading and Initialization
print("Loading spatial data...")
adata = sc.read_visium(sp_data_folder, count_file=count_file, load_images=True)
adata.var_names_make_unique()

print(f"\nData dimensions: {adata.n_obs} spots, {adata.n_vars} genes")
print("\nObservation metadata:")
print(adata.obs.head())
print("\nVariable metadata:")
print(adata.var.head())

# 2. Quality Control and Preprocessing


def perform_qc(adata):
    """Perform quality control and preprocessing."""
    print("\nPerforming quality control...")

    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    ribo_genes = pd.read_csv('/home/student/Documents/python-project/KEGG_RIBOSOME.v2024.1.Hs.csv',
                             skiprows=2, header=None)
    adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True)

    # Print QC summary
    print(f"\nNumber of spots under tissue: {adata.n_obs}")
    print(f"Mean reads per spot: {adata.obs['total_counts'].mean():.1f}")
    print(f"Median genes per spot: {adata.obs['n_genes_by_counts'].median()}")

    # Plot QC metrics
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0], bins=50)
    axs[0].set_title("Total Counts Distribution")
    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=50, ax=axs[1])
    axs[1].set_title("Detected Genes Distribution")
    plt.tight_layout()
    plt.show()

    # Violin plots of QC metrics
    sc.pl.violin(
        adata,
        ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo'],
        jitter=0.4,
        multi_panel=True,
        stripplot=False,
        title="QC Metrics Distribution"
    )

    return adata


adata = perform_qc(adata)

# 3. Data Filtering


def filter_data(adata):
    """Filter cells and genes based on QC metrics."""
    print("\nFiltering data...")

    # Cell filtering
    print(f"Before filtering: {adata.n_obs} cells")
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)
    adata = adata[adata.obs["pct_counts_mt"] < 20]
    adata = adata[adata.obs["pct_counts_ribo"] < 2]
    print(f"After filtering: {adata.n_obs} cells")

    # Gene filtering
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"Genes remaining: {adata.n_vars}")

    return adata


adata = filter_data(adata)

# 4. Visualization Functions


def plot_top_genes(adata, n_genes=20, log_transform=True):
    """Plot top expressed genes with enhanced visualization."""
    X_dense = adata.X.toarray() if issparse(adata.X) else adata.X

    if log_transform:
        X_dense = np.log2(X_dense + 1)

    gene_means = np.mean(X_dense, axis=0)
    top_indices = np.argsort(gene_means)[::-1][:n_genes]
    top_genes = adata.var_names[top_indices]
    top_values = gene_means[top_indices]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_genes, top_values,
                   color=plt.cm.viridis(np.linspace(0, 1, n_genes)))

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.title(
        f"Top {n_genes} {'Log2 ' if log_transform else ''}Expressed Genes")
    plt.ylabel(f"Mean Expression{' (log2)' if log_transform else ''}")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_gene_histogram(adata, gene_name):
    """Plot expression distribution for a specific gene."""
    if gene_name not in adata.var_names:
        print(f"Gene {gene_name} not found in dataset")
        return

    gene_index = list(adata.var_names).index(gene_name)
    counts = adata.X[:, gene_index].toarray().flatten()

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Raw counts
    ax[0].hist(counts, bins=30, color='skyblue', edgecolor='black')
    ax[0].set_title(f"Raw Counts for {gene_name}")
    ax[0].set_xlabel("Counts per spot")
    ax[0].set_ylabel("Frequency")

    # Normalized counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    norm_counts = adata.X[:, gene_index].toarray().flatten()
    ax[1].hist(norm_counts, bins=30, color='salmon', edgecolor='black')
    ax[1].set_title(f"Normalized Counts for {gene_name}")
    ax[1].set_xlabel("Normalized counts per spot")

    # Log-transformed counts
    sc.pp.log1p(adata)
    log_counts = adata.X[:, gene_index].toarray().flatten()
    ax[2].hist(log_counts, bins=30, color='lightgreen', edgecolor='black')
    ax[2].set_title(f"Log-Transformed Counts for {gene_name}")
    ax[2].set_xlabel("Log(counts+1) per spot")

    plt.tight_layout()
    plt.show()


# Visualize top genes
plot_top_genes(adata, n_genes=20, log_transform=True)

# Example gene visualization
plot_gene_histogram(adata, 'mt-Co3')

# 5. Spatial Visualization


def plot_spatial_features(adata, features, ncols=2, **kwargs):
    """Enhanced spatial plotting function."""
    default_kwargs = {
        'img_key': 'hires',
        'size': 1.5,
        'alpha_img': 0.6,
        'frameon': False,
        'cmap': 'magma',
        'vmin': 0,
        'vmax': 'p99',
        'ncols': ncols,
        'wspace': 0.3,
        'colorbar_loc': 'right'
    }
    default_kwargs.update(kwargs)

    sc.pl.spatial(adata, color=features, **default_kwargs)


# Plot spatial features
plot_spatial_features(adata, ["total_counts", "n_genes_by_counts"])
plot_spatial_features(adata, ["Rorb", "Vip"], ncols=2)

# 6. Dimensionality Reduction and Clustering


def analyze_clusters(adata, resolution=0.6):
    """Perform clustering analysis with visualization."""
    print("\nPerforming clustering analysis...")

    # Normalization and HVG selection
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    sc.pl.highly_variable_genes(adata)
    adata = adata[:, adata.var.highly_variable]

    # Dimensionality reduction
    sc.pp.pca(adata)
    sc.pl.pca_variance_ratio(adata, log=True)

    # Clustering
    sc.pp.neighbors(adata, n_pcs=20)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="clusters")

    # Visualization
    sc.pl.umap(adata, color=["clusters", "total_counts", "n_genes_by_counts"],
               wspace=0.4, frameon=False)

    # Spatial visualization of clusters
    plot_spatial_features(adata, "clusters", cmap='tab20', size=1.2)

    return adata


adata = analyze_clusters(adata, resolution=0.6)

# 7. Marker Gene Analysis


def find_markers(adata, method='t-test', n_genes=10):
    """Perform differential expression analysis."""
    print("\nIdentifying marker genes...")
    sc.tl.rank_genes_groups(adata, "clusters", method=method)

    # Plot results
    sc.pl.rank_genes_groups(adata, n_genes=n_genes, sharey=False)

    # Print top markers for each cluster
    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names

    print("\nTop marker genes per cluster:")
    for group in groups:
        markers = result["names"][group][:n_genes]
        print(f"Cluster {group}: {', '.join(markers)}")

    # Plot heatmap for specific cluster
    sc.pl.rank_genes_groups_heatmap(
        adata,
        groups=groups[:3],  # First 3 clusters
        n_genes=n_genes,
        groupby="clusters",
        show_gene_labels=True,
        dendrogram=True
    )

    return adata


adata = find_markers(adata)

# 8. Spatial Visualization of Marker Genes
top_markers = []
for group in adata.uns["rank_genes_groups"]["names"].dtype.names:
    top_markers.append(adata.uns["rank_genes_groups"]["names"][group][0])

plot_spatial_features(adata, top_markers, ncols=3, size=1.5, cmap='viridis')

# 9. Save Results
adata.write('processed_spatial_data.h5ad', compression='gzip')
print("\nAnalysis complete. Results saved to processed_spatial_data.h5ad")
