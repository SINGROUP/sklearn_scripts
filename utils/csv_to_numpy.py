import numpy as np
import argparse

### INPUT ###
parser = argparse.ArgumentParser(description='csv filename.')

parser.add_argument('arguments', metavar='cla', type=str, nargs='+',
                   help='[give csv file in (data*features)-format ]')

args = parser.parse_args()
print("Arguments passed:")
print(args.arguments)

infile = args.arguments[0]

csv = np.loadtxt(infile)

np.save(infile + ".npy", csv)
