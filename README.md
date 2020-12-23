# PASTE

PASTE is a computational method that leverages both gene expression similarity and spatial distances between spots align and integrate spatial transcriptomics data. In particular, there are two methods:
1. `pairwise_align`: align spots across pairwise ST layers.
2. `center_align`: integrate multiple ST layers into one center layer.

PASTE is actively being worked on with future updates coming. 

# Updates

### Command Line

You can now run PASTE from the command line. 

Sample execution: `python paste.py -m pairwise -f file1.csv file2.csv file3.csv`

Note: `pairwise` will return pairwise alignment between each consecutive pair of files (e.g. \[file1,file2\], \[file2,file3\]).

| Flag | Name | Description | Default Value |
| --- | --- | --- | --- |
| -m | mode | Select either `pairwise` or `center` | (str) `pairwise` |
| -f | files | Path to data files (.csv) | None |
| -d | direc | Directory to store output files | Current Directory |
| -a | alpha | alpha parameter for PASTE | (float) `0.1` |
| -p | n_components | n_components for NMF step in `center_align` | (int) `15` |
| -l | lmbda | lambda parameter in `center_align` | (floats) probability vector of length `n`  |
| -i | intial_layer | Specify which file is also the intial layer in `center_align` | (int) `1` |
| -t | threshold | Convergence threshold for `center_align` | (float) `0.001` |

Input files are .csv files of the form:

```
       	'gene_a'  'gene_b'
'2x5'	   0         9      
'2x7'	   2         6      
```
Where the columns indexes are gene names (str), row indexes are spatial coordinates (str), and entries are gene counts (int). In particular, row indexes are of the form `AxB` where `A` and `B` are floats.

### Sample Dataset

Added sample spatial transcriptomics dataset consisting of four breast cancer layers courtesy of:

Ståhl, Patrik & Salmén, Fredrik & Vickovic, Sanja & Lundmark, Anna & Fernandez Navarro, Jose & Magnusson, Jens & Giacomello, Stefania & Asp, Michaela & Westholm, Jakub & Huss, Mikael & Mollbrink, Annelie & Linnarsson, Sten & Codeluppi, Simone & Borg, Åke & Pontén, Fredrik & Costea, Paul & Sahlén, Pelin Akan & Mulder, Jan & Bergmann, Olaf & Frisén, Jonas. (2016). Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. Science. 353. 78-82. 10.1126/science.aaf2403. 

Note: Original data is (.tsv), but we converted it to (.csv).
