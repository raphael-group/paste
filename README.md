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

| Flag | Name | Description |
| --- | --- | ---|
| -m | mode | Select either `pairwise` or `center` |
| -f | files | Path to data files (.csv) |
| -d | direc | Directory to store output files |
| -a | alpha | alpha parameter for PASTE |
| -p | n_components | n_components for NMF step in `center_align` |
| -t | threshold | Convergence threshold for `center_align` |

Input files are .csv files of the form:

```
       	'gene_a'  'gene_b'
'2x5'	   0         9      
'2x7'	   2         6      
```
Where the columns indexes are gene names, row indexes are spatial coordinates, and entries are gene counts.

### Sample Dataset

Added sample spatial transcriptomics dataset consisting of four breast cancer layers courtesy of:

Ståhl, Patrik & Salmén, Fredrik & Vickovic, Sanja & Lundmark, Anna & Fernandez Navarro, Jose & Magnusson, Jens & Giacomello, Stefania & Asp, Michaela & Westholm, Jakub & Huss, Mikael & Mollbrink, Annelie & Linnarsson, Sten & Codeluppi, Simone & Borg, Åke & Pontén, Fredrik & Costea, Paul & Sahlén, Pelin Akan & Mulder, Jan & Bergmann, Olaf & Frisén, Jonas. (2016). Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. Science. 353. 78-82. 10.1126/science.aaf2403. 
