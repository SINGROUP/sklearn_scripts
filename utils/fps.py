# algorithms taken from
# https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
# forum as of 21.03.2018

import numpy as np
from time import clock
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import sys

def timeit(func):
    def wrapper(*args, **kwargs):
        starting_time = clock()
        result = func(*args, **kwargs)
        ending_time = clock()
        print('Duration: {}'.format(ending_time - starting_time))
        return result
    return wrapper

def dist_ponto_cj(ponto,lista):
    return [ euclidean(ponto,lista[j]) for j in range(len(lista)) ]

def ponto_mais_longe(lista_ds):
    ds_max = max(lista_ds)
    idx = lista_ds.index(ds_max)
    return pts[idx]

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

# slow algorithm
@timeit
def op(pts, N, K):
    farthest_pts = [0] * K

    P0 = pts[np.random.randint(0, N)]
    farthest_pts[0] = P0
    ds0 = dist_ponto_cj(P0, pts)

    ds_tmp = ds0
    for i in range(1, K):
        farthest_pts[i] = ponto_mais_longe(ds_tmp)
        ds_tmp2 = dist_ponto_cj(farthest_pts[i], pts)
        ds_tmp = [min(ds_tmp[j], ds_tmp2[j]) for j in range(len(ds_tmp))]
        # print ('P[%d]: %s' % (i, farthest_pts[i]))
    return farthest_pts

# fast algorithm
@timeit
def graipher(pts, K):
    dim = np.shape(pts)[1]
    farthest_pts = np.zeros((K, dim))
    farthest_ids = np.zeros(K) 
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_ids[i] = np.argmax(distances)
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, farthest_ids




if __name__ == "__main__":
    filename = sys.argv[1]  #soap
    #"test4999_soap_n10l9r5.npy"

    #for K in [250,300,350,500,1000,1500,2000]:
    for K in [2000]:
        print("searching for", str(K), "farthest points")
        pts = np.load(filename)

        farthest_pts, farthest_ids = graipher(pts, K)
        """
        fig, ax = plt.subplots()
        plt.grid(False)
        plt.scatter(pts[:, 0], pts[:, 1], c='k', s=4)
        #plt.scatter(xf, yf, c='r', s=4)
        plt.scatter(farthest_pts[:,0], farthest_pts[:,1],c='r',s=4)
        plt.show()
        """
        np.save("fps_" + str(K) + "ids_" + filename, farthest_ids.astype('int'))
        np.save("fps_" + str(K) + "_" + filename, farthest_pts)

    N = 2200
    restids = np.setdiff1d(np.arange(0,N), farthest_ids)
    print(restids.shape)
    np.save("fps_" + str(K) + "restids_" + filename, restids.astype('int'))
    np.save("fps_" + str(K) + "rest_" + filename, pts[restids])

    print('done')
