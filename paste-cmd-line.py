import math
import scanpy as sc
import numpy as np
import pandas as pd
import argparse
import os
from src.paste import pairwise_align, center_align

def main(args):
    print(args)
    n_slices = int(len(args.filename)/2)
    # Error check arguments
    if args.mode != 'pairwise' and args.mode != 'center':
        print("Error: please select either 'pairwise' or 'center' mode.")
        return
    
    if args.alpha < 0 or args.alpha > 1:
        print("Error: alpha specified outside [0, 1].")
        return
    
    if args.initial_slice < 1 or args.initial_slice > n_slices:
        print("Error: initial slice specified outside [1, n].")
        return
    
    if len(args.lmbda) == 0:
        lmbda = n_slices*[1./n_slices]
    elif len(args.lmbda) != n_slices:
        print("Error: length of lambda does not equal number of files.")
        return
    else:
        if not all(i >= 0 for i in args.lmbda):
            print("Error: lambda includes negative weights.")
            return
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
    
    # create output folder
    output_path = os.path.join(args.direc, "paste_output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    if args.mode == 'pairwise':
        print("Computing pairwise alignment.")
        # compute pairwise align
        for i in range(n_slices - 1):
            pi = pairwise_align(slices[i], slices[i+1], args.alpha)
            pi = pd.DataFrame(pi, index = slices[i].obs.index, columns = slices[i+1].obs.index)
            output_filename = "paste_output/slice" + str(i+1) + "_slice" + str(i+2) + "_pairwise.csv"
            pi.to_csv(os.path.join(args.direc, output_filename))
        return
    elif args.mode == 'center':
        print("Computing center alignment.")
        initial_slice = slices[args.initial_slice - 1].copy()
        # compute center align
        center_slice, pis = center_align(initial_slice, slices, lmbda, args.alpha, args.n_components, args.threshold)
        W = pd.DataFrame(center_slice.uns['paste_W'], index = center_slice.obs.index)
        H = pd.DataFrame(center_slice.uns['paste_H'], columns = center_slice.var.index)
        W.to_csv(os.path.join(args.direc,"paste_output/W_center"))
        H.to_csv(os.path.join(args.direc,"paste_output/H_center"))
        for i in range(len(pis)):
            output_filename = "paste_output/slice_center_slice" + str(i+1) + "_pairwise"
            pi = pd.DataFrame(pis[i], index = center_slice.obs.index, columns = slices[i].obs.index)
            pi.to_csv(os.path.join(args.direc, output_filename))
        return
    return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename", help="path to data files (.csv). Alternate between gene expression and spatial data. Ex: slice1_gene.csv, slice1_coor.csv, slice2_gene.csv, slice2_coor.csv",type=str, default=[], nargs='+')
    parser.add_argument("-m","--mode", help="either 'pairwise' or 'center' ", type=str, default="pairwise")
    parser.add_argument("-d","--direc", help="directory to save files",default='')
    parser.add_argument("-a","--alpha", help="alpha param for PASTE (float from [0,1])",type=float, default = 0.1)
    parser.add_argument("-p","--n_components", help="n_components for NMF step in center_align",type=int, default = 15)
    parser.add_argument("-l", "--lmbda", help="lambda param in center_align (weight vector of length n) ",type=float, default = [], nargs='+') 
    parser.add_argument("-i", "--initial_slice", help="specify which slice is the intial slice for center_align (int from 1-n)",type=int, default = 1) 
    parser.add_argument("-t","--threshold", help="convergence threshold for center_align",type=float, default = 0.001)
    args = parser.parse_args()
    main(args)
