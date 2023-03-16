[![PyPI](https://img.shields.io/pypi/v/paste-bio.svg)](https://pypi.org/project/paste-bio)
[![Downloads](https://pepy.tech/badge/paste-bio)](https://pepy.tech/project/paste-bio)
[![Documentation Status](https://readthedocs.org/projects/paste-bio/badge/?version=latest)](https://paste-bio.readthedocs.io/en/stable/?badge=stable)
[![Anaconda](https://anaconda.org/bioconda/paste-bio/badges/version.svg)](https://anaconda.org/bioconda/paste-bio/badges/version.svg)
[![bioconda-downloads](https://anaconda.org/bioconda/paste-bio/badges/downloads.svg)](https://anaconda.org/bioconda/paste-bio/badges/downloads.svg)

# PASTE

![PASTE Overview](https://github.com/raphael-group/paste/blob/main/docs/source/_static/images/paste_overview.png)

PASTE is a computational method that leverages both gene expression similarity and spatial distances between spots to align and integrate spatial transcriptomics data. In particular, there are two methods:
1. `pairwise_align`: align spots across pairwise slices.
2. `center_align`: integrate multiple slices into one center slice.

You can read full paper [here](https://www.nature.com/articles/s41592-022-01459-6). 

Additional examples and the code to reproduce the paper's analyses can be found [here](https://github.com/raphael-group/paste_reproducibility). Preprocessed datasets used in the paper can be found on [zenodo](https://doi.org/10.5281/zenodo.6334774).

### Recent News

* PASTE is now published in [Nature Methods](https://www.nature.com/articles/s41592-022-01459-6)!

* The code to reproduce the analisys can be found [here](https://github.com/raphael-group/paste_reproducibility).

* As of version 1.2.0, PASTE now supports GPU implementation via Pytorch. For more details, see the GPU section of the [Tutorial notebook](docs/source/notebooks/getting-started.ipynb).

### Installation

The easiest way is to install PASTE on pypi: https://pypi.org/project/paste-bio/. 

`pip install paste-bio` 

Or you can install PASTE on bioconda: https://anaconda.org/bioconda/paste-bio.

`conda install -c bioconda paste-bio`

Check out Tutorial.ipynb for an example of how to use PASTE.

Alternatively, you can clone the respository and try the following example in a
notebook or the command line. 

### Quick Start

To use PASTE we require at least two slices of spatial-omics data (both
expression and coordinates) that are in
anndata format (i.e. read in by scanpy/squidpy). We have included a breast
cancer dataset from [1] in the [sample_data folder](sample_data/) of this repo 
that we will use as an example below to show how to use PASTE.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import paste as pst

# Load Slices
data_dir = './sample_data/' # change this path to the data you wish to analyze

# Assume that the coordinates of slices are named slice_name + "_coor.csv"
def load_slices(data_dir, slice_names=["slice1", "slice2"]):
    slices = []  
    for slice_name in slice_names:
        slice_i = sc.read_csv(data_dir + slice_name + ".csv")
        slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
        slice_i.obsm['spatial'] = slice_i_coor
        # Preprocess slices
        sc.pp.filter_genes(slice_i, min_counts = 15)
        sc.pp.filter_cells(slice_i, min_counts = 100)
        slices.append(slice_i)
    return slices

slices = load_slices(data_dir)
slice1, slice2 = slices

# Pairwise align the slices
pi12 = pst.pairwise_align(slice1, slice2)

# To visualize the alignment you can stack the slices 
# according to the alignment pi
slices, pis = [slice1, slice2], [pi12]
new_slices = pst.stack_slices_pairwise(slices, pis)

slice_colors = ['#e41a1c','#377eb8']
plt.figure(figsize=(7,7))
for i in range(len(new_slices)):
    pst.plot_slice(new_slices[i],slice_colors[i],s=400)
plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='1'),mpatches.Patch(color=slice_colors[1], label='2')])
plt.gca().invert_yaxis()
plt.axis('off')
plt.show()

# Center align slices
## We have to reload the slices as pairwise_alignment modifies the slices.
slices = load_slices(data_dir)
slice1, slice2 = slices

# Construct a center slice
## choose one of the slices as the coordinate reference for the center slice,
## i.e. the center slice will have the same number of spots as this slice and
## the same coordinates.
initial_slice = slice1.copy()    
slices = [slice1, slice2]
lmbda = len(slices)*[1/len(slices)] # set hyperparameter to be uniform

## Possible to pass in an initial pi (as keyword argument pis_init) 
## to improve performance, see Tutorial.ipynb notebook for more details.
center_slice, pis = pst.center_align(initial_slice, slices, lmbda) 

## The low dimensional representation of our center slice is held 
## in the matrices W and H, which can be used for downstream analyses
W = center_slice.uns['paste_W']
H = center_slice.uns['paste_H']
```

### GPU implementation
PASTE now is compatible with gpu via Pytorch. All we need to do is add the following two parameters to our main functions:
```
pi12 = pst.pairwise_align(slice1, slice2, backend = ot.backend.TorchBackend(), use_gpu = True)

center_slice, pis = pst.center_align(initial_slice, slices, lmbda, backend = ot.backend.TorchBackend(), use_gpu = True) 
```
For more details, see the GPU section of the [Tutorial notebook](docs/source/notebooks/getting-started.ipynb).

### Command Line

We provide the option of running PASTE from the command line. 

First, clone the repository:

`git clone https://github.com/raphael-group/paste.git`

Next, when providing files, you will need to provide two separate files: the gene expression data followed by spatial data (both as .csv) for the code to initialize one slice object.

Sample execution (based on this repo): `python paste-cmd-line.py -m center -f ./sample_data/slice1.csv ./sample_data/slice1_coor.csv ./sample_data/slice2.csv ./sample_data/slice2_coor.csv ./sample_data/slice3.csv ./sample_data/slice3_coor.csv`

Note: `pairwise` will return pairwise alignment between each consecutive pair of slices (e.g. \[slice1,slice2\], \[slice2,slice3\]).

| Flag | Name | Description | Default Value |
| --- | --- | --- | --- |
| -m | mode | Select either `pairwise` or `center` | (str) `pairwise` |
| -f | files | Path to data files (.csv) | None |
| -d | direc | Directory to store output files | Current Directory |
| -a | alpha | Alpha parameter for PASTE | (float) `0.1` |
| -c | cost | Expression dissimilarity cost (`kl` or `Euclidean`) | (str) `kl` |
| -p | n_components | n_components for NMF step in `center_align` | (int) `15` |
| -l | lmbda | Lambda parameter in `center_align` | (floats) probability vector of length `n`  |
| -i | intial_slice | Specify which file is also the intial slice in `center_align` | (int) `1` |
| -t | threshold | Convergence threshold for `center_align` | (float) `0.001` |
| -x | coordinates | Output new coordinates (toggle to turn on) | `False` |
| -w | weights | Weights files of spots in each slice (.csv) | None |
| -s | start | Initial alignments for OT. If not given uses uniform (.csv structure similar to alignment output) | None |

`pairwise_align` outputs a (.csv) file containing mapping of spots between each consecutive pair of slices. The rows correspond to spots of the first slice, and cols the second.

`center_align` outputs two files containing the low dimensional representation (NMF decomposition) of the center slice gene expression, and files containing a mapping of spots between the center slice (rows) to each input slice (cols).

### Sample Dataset

Added sample spatial transcriptomics dataset consisting of four breast cancer slice courtesy of:

[1] Ståhl, Patrik & Salmén, Fredrik & Vickovic, Sanja & Lundmark, Anna & Fernandez Navarro, Jose & Magnusson, Jens & Giacomello, Stefania & Asp, Michaela & Westholm, Jakub & Huss, Mikael & Mollbrink, Annelie & Linnarsson, Sten & Codeluppi, Simone & Borg, Åke & Pontén, Fredrik & Costea, Paul & Sahlén, Pelin Akan & Mulder, Jan & Bergmann, Olaf & Frisén, Jonas. (2016). Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. Science. 353. 78-82. 10.1126/science.aaf2403. 

Note: Original data is (.tsv), but we converted it to (.csv).

### References

Ron Zeira, Max Land, Alexander Strzalkowski and Benjamin J. Raphael. "Alignment and integration of spatial transcriptomics data". Nature Methods (2022). https://doi.org/10.1038/s41592-022-01459-6
