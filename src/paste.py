import math
import pandas as pd
import numpy as np
import argparse
import os
from STLayer import STLayer
from PASTE import pairwise_align, center_align
from helper import getCoordinates

def main(args):
    print(args)
    # Error check arguments
    if args.mode != 'pairwise' and args.mode != 'center':
        print("Please select either 'pairwise' or 'center' mode")
        return
    
    # create STLayers
    n_layers = len(args.filename)
    layers = []
    for i in range(n_layers):
        df = pd.read_csv(args.filename[i], sep="\t", index_col=0)
        layers.append(STLayer(df, getCoordinates(df)))
    
    # create output folder
    os.mkdir(os.path.join(args.direc, "paste_output"))
    
    if args.mode == 'pairwise':
        # compute pairwise align
        for i in range(n_layers - 1):
            pi = pairwise_align(layers[i], layers[i+1], args.alpha)
            output_filename = "paste_output/layer" + str(i+1) + "_layer" + str(i+2) + "_pairwise"
            np.savetxt(os.path.join(args.direc, output_filename), pi, delimiter=',')
        return
    elif args.mode == 'center':
        lmbda = n_layers*[1/n_layers]
        initial_layer = layers[0]
        # compute center align
        W, H = center_align(initial_layer, layers, lmbda, args.alpha, args.n_components, args.threshold)
        np.savetxt(os.path.join(args.direc,"paste_output/W_center"), W, delimiter=',')
        np.savetxt(os.path.join(args.direc,"paste_output/H_center"), H, delimiter=',')
        return
    return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename", help="path to data file (.tsv)",type=str, default=[], nargs='+')
    parser.add_argument("-m","--mode", help="either 'pairwise' or 'center' ", type=str, default="pairwise")
    parser.add_argument("-d","--direc", help="directory to save files",default='')
    parser.add_argument("-a","--alpha", help="alpha param for PASTE",type=float, default = 0.1)
    parser.add_argument("-p","--n_components", help="n_components for NMF",type=int, default = 15)
    parser.add_argument("-t","--threshold", help="convergence threshold for center align",type=float, default = 0.001)
    args = parser.parse_args()
    main(args)
