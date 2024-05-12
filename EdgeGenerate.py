import numpy as np
import matplotlib.pyplot as plt
import random


def EdgeGene():
    THRESHOLD = 200
    A = np.zeros((21, 21))
    rng = np.random.default_rng()
    # print(rng.random(3))

    pos = rng.integers(0, 200, size=(21, 2), endpoint=True)
    pos[0] = [100, 100]
    # print(pos)

    # plt.scatter(pos[0, 0], pos[0, 1], marker='*')
    # plt.scatter(pos[1:, 0], pos[1:, 1])

    for i in range(21):
        for j in range(i+1, 21):
            tmp = pos[i] - pos[j]
            dis = np.sqrt((tmp*tmp).sum())
            if (dis < THRESHOLD):
                A[i, j] = A[j, i] = 1 / dis

    Su_Su_Distance = A[1:, 1:]  # dim = (20,20)
    PU_SU_Distance = np.zeros((1, 20))
    EdgeMatrix = np.concatenate((PU_SU_Distance, Su_Su_Distance), axis=0)
    return EdgeMatrix


if __name__ == "__main__":
    EdgeMatrix = EdgeGene()
    # print(A.shape)
    # print(Su_Su_Distance.shape)
    print("EdgeMatrix shape = ", EdgeMatrix.shape)

    plt.show()
