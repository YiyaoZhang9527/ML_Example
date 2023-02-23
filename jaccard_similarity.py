import numpy as np


def jaccard_similarity(setA, setB):
    return np.intersect1d(setA, setB).shape[0]/np.union1d(setA, setB).shape[0]

if __name__ == "__main__":
    A, B = np.arange(10), np.arange(5, 12)
    print(jaccard_similarity(A, B))
