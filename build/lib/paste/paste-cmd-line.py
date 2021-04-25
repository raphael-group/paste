import math
import pandas as pd
import numpy as np
import argparse
import os
from .STLayer import STLayer
from .PASTE import pairwise_align, center_align
from .hepler import getCoordinates

def main(args):
    print(args)
    n_layers = len(args.filename)
    # Error check arguments
    if args.mode != 'pairwise' and args.mode != 'center':
        print("Error: please select either 'pairwise' or 'center' mode.")
        return
    
    if args.alpha < 0 or args.alpha > 1:
        print("Error: alpha specified outside [0, 1].")
        return
    
    if args.initial_layer < 1 or args.initial_layer > n_layers:
        print("Error: initial layer specified outside [1, n].")
        return
    
    if len(args.lmbda) == 0:
        lmbda = n_layers*[1/n_layers]
    elif len(args.lmbda) != n_layers:
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
        
    # create STLayers
    layers = []
    for i in range(n_layers):
        df = pd.read_csv(args.filename[i], index_col=0)
        layers.append(STLayer(df, getCoordinates(df)))
    
    # create output folder
    output_path = os.path.join(args.direc, "paste_output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    if args.mode == 'pairwise':
        print("Computing pairwise alignment.")
        # compute pairwise align
        for i in range(n_layers - 1):
            pi = pairwise_align(layers[i], layers[i+1], args.alpha)
            output_filename = "paste_output/layer" + str(i+1) + "_layer" + str(i+2) + "_pairwise"
            np.savetxt(os.path.join(args.direc, output_filename), pi, delimiter=',')
        return
    elif args.mode == 'center':
        print("Computing center alignment.")
        initial_layer = layers[args.initial_layer - 1]
        # compute center align
        W, H, pis = center_align(initial_layer, layers, lmbda, args.alpha, args.n_components, args.threshold)
        np.savetxt(os.path.join(args.direc,"paste_output/W_center"), W, delimiter=',')
        np.savetxt(os.path.join(args.direc,"paste_output/H_center"), H, delimiter=',')
        for i in range(len(pis)):
            output_filename = "paste_output/layer_center_layer" + str(i+1) + "_pairwise"
            np.savetxt(os.path.join(args.direc, output_filename), pis[i], delimiter=',')
        return
    return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename", help="path to data file (.csv)",type=str, default=[], nargs='+')
    parser.add_argument("-m","--mode", help="either 'pairwise' or 'center' ", type=str, default="pairwise")
    parser.add_argument("-d","--direc", help="directory to save files",default='')
    parser.add_argument("-a","--alpha", help="alpha param for PASTE (float from [0,1])",type=float, default = 0.1)
    parser.add_argument("-p","--n_components", help="n_components for NMF step in center_align",type=int, default = 15)
    parser.add_argument("-l", "--lmbda", help="lambda param in center_align (weight vector of length n) ",type=float, default = [], nargs='+') 
    parser.add_argument("-i", "--initial_layer", help="specify which file is the intial layer for center_align (int from 1-n)",type=int, default = 1) 
    parser.add_argument("-t","--threshold", help="convergence threshold for center_align",type=float, default = 0.001)
    args = parser.parse_args()
    main(args)
