# PASTE

PASTE is a computational method that leverages both gene expression similarity and spatial distances between spots align and integrate spatial transcriptomics data. In particular, there are two methods:
1. `pairwise_align`: align spots across pairwise ST layers.
2. `center_align`: integrate multiple ST layers into one center layer.

PASTE is actively being worked on with future updates coming. 

# Updates

You can now run PASTE from the command line. 

Sample execution: `python paste.py -m pairwise -f file1.tsv file2.tsv file3.tsv`
