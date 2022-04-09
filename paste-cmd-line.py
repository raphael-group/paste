import math
import scanpy as sc
import numpy as np
import pandas as pd
import argparse
import os
from src.paste import pairwise_align, center_align, stack_slices_pairwise, stack_slices_center

def main(args):
    # print(args)
    n_slices = int(len(args.filename)/2)
    # Error check arguments
    if args.mode != 'pairwise' and args.mode != 'center':
        raise(ValueError("Please select either 'pairwise' or 'center' mode."))
    
    if args.alpha < 0 or args.alpha > 1:
        raise(ValueError("alpha specified outside [0, 1]"))
    
    if args.initial_slice < 1 or args.initial_slice > n_slices:
        raise(ValueError("Initial slice specified outside [1, n]"))
    
    if len(args.lmbda) == 0:
        lmbda = n_slices*[1./n_slices]
    elif len(args.lmbda) != n_slices:
        raise(ValueError("Length of lambda does not equal number of files"))
    else:
        if not all(i >= 0 for i in args.lmbda):
            raise(ValueError("lambda includes negative weights"))
        else:
            print("Normalizing lambda weights into probability vector.")
            lmbda = args.lmbda
            lmbda = [float(i)/sum(lmbda) for i in lmbda]

    # create slices
    slices = []
    for i in range(n_slices):
        s = sc.read_csv(args.filename[2*i])
        s.obsm['spatial'] = np.genfromtxt(args.filename[2*i+1], delimiter = ',')
        slices.append(s)

    if len(args.weights)==0:
        for i in range(n_slices):
            slices[i].obsm['weights'] = np.ones((slices[i].shape[0],))/slices[i].shape[0]
    elif len(args.weights)!=n_slices:
        raise(ValueError("Number of slices {0} != number of weight files {1}".format(n_slices,len(args.weights))))
    else:
        for i in range(n_slices):
            slices[i].obsm['weights'] = np.genfromtxt(args.weights[i], delimiter = ',')
            slices[i].obsm['weights'] = slices[i].obsm['weights']/np.sum(slices[i].obsm['weights'])
    
    if len(args.start)==0:
        pis_init = (n_slices-1)*[None] if args.mode == 'pairwise' else None
    elif (args.mode == 'pairwise' and len(args.start)!=n_slices-1) or (args.mode == 'center' and len(args.start)!=n_slices):
        raise(ValueError("Number of slices {0} != number of start pi files {1}".format(n_slices,len(args.start))))
    else:
        pis_init = [pd.read_csv(args.start[i],index_col=0).to_numpy() for i in range(len(args.start))]

    # create output folder
    output_path = os.path.join(args.direc, "paste_output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    if args.mode == 'pairwise':
        print("Computing pairwise alignment.")
        # compute pairwise align
        pis = []
        for i in range(n_slices - 1):
            pi = pairwise_align(slices[i], slices[i+1], args.alpha, dissimilarity=args.cost, a_distribution=slices[i].obsm['weights'], b_distribution=slices[i+1].obsm['weights'], G_init=pis_init[i])
            pis.append(pi)
            pi = pd.DataFrame(pi, index = slices[i].obs.index, columns = slices[i+1].obs.index)
            output_filename = "paste_output/slice" + str(i+1) + "_slice" + str(i+2) + "_pairwise.csv"
            pi.to_csv(os.path.join(args.direc, output_filename))
        if args.coordinates:
            new_slices = stack_slices_pairwise(slices, pis)
            for i in range(n_slices):
                output_filename = "paste_output/slice" + str(i+1) + "_new_coordinates.csv"
                np.savetxt(os.path.join(args.direc, output_filename), new_slices[i].obsm['spatial'], delimiter=",")
    elif args.mode == 'center':
        print("Computing center alignment.")
        initial_slice = slices[args.initial_slice - 1].copy()
        # compute center align
        center_slice, pis = center_align(initial_slice, slices, lmbda, args.alpha, args.n_components, args.threshold, dissimilarity=args.cost, distributions=[slices[i].obsm['weights'] for i in range(n_slices)], pis_init=pis_init)
        W = pd.DataFrame(center_slice.uns['paste_W'], index = center_slice.obs.index)
        H = pd.DataFrame(center_slice.uns['paste_H'], columns = center_slice.var.index)
        W.to_csv(os.path.join(args.direc,"paste_output/W_center"))
        H.to_csv(os.path.join(args.direc,"paste_output/H_center"))
        for i in range(len(pis)):
            output_filename = "paste_output/slice_center_slice" + str(i+1) + "_pairwise.csv"
            pi = pd.DataFrame(pis[i], index = center_slice.obs.index, columns = slices[i].obs.index)
            pi.to_csv(os.path.join(args.direc, output_filename))
        if args.coordinates:
            center, new_slices = stack_slices_center(center_slice, slices, pis)
            for i in range(n_slices):
                output_filename = "paste_output/slice" + str(i+1) + "_new_coordinates.csv"
                np.savetxt(os.path.join(args.direc, output_filename), new_slices[i].obsm['spatial'], delimiter=",")
            np.savetxt(os.path.join(args.direc, "paste_output/center_new_coordinates.csv"), center.obsm['spatial'], delimiter=",")
    return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename", help="path to data files (.csv). Alternate between gene expression and spatial data. Ex: slice1_gene.csv, slice1_coor.csv, slice2_gene.csv, slice2_coor.csv",type=str, default=[], nargs='+')
    parser.add_argument("-m","--mode", help="either 'pairwise' or 'center' ", type=str, default="pairwise")
    parser.add_argument("-d","--direc", help="directory to save files",default='')
    parser.add_argument("-a","--alpha", help="alpha param for PASTE (float from [0,1])",type=float, default = 0.1)
    parser.add_argument("-c","--cost", help="expression dissimilarity cost, either 'kl' or 'euclidean' ", type=str, default="kl")
    parser.add_argument("-p","--n_components", help="n_components for NMF step in center_align",type=int, default = 15)
    parser.add_argument("-l", "--lmbda", help="lambda param in center_align (weight vector of length n) ",type=float, default = [], nargs='+') 
    parser.add_argument("-i", "--initial_slice", help="specify which slice is the intial slice for center_align (int from 1-n)",type=int, default = 1) 
    parser.add_argument("-t","--threshold", help="convergence threshold for center_align",type=float, default = 0.001)
    parser.add_argument("-x","--coordinates", help="output new coordinates", action='store_true', default = False)
    parser.add_argument("-w","--weights", help="path to files containing weights of spots in each slice. The format of the files is the same as the coordinate files used as input",type=str, default=[], nargs='+')
    parser.add_argument("-s","--start", help="path to files containing initial starting alignmnets. If not given the OT starts the search with uniform alignments. The format of the files is the same as the alignments files output by PASTE",type=str, default=[], nargs='+')
    args = parser.parse_args()
    main(args)
