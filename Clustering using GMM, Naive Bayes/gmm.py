import json
import random
import numpy as np


def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []

    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###
    X = np.array(X)
    mu = np.array(mu)
    cov = np.array(cov)
    gamma = np.zeros((X.shape[0],K))
    # Run 100 iterations of EM updates
    for t in range(100):
        for k in range(K):
            exp = np.exp(-0.5 * np.sum(np.multiply((X - mu[k]), (np.dot(np.linalg.inv(cov[k].reshape(2, 2)), (X - mu[k]).T)).T), axis=1))
            con = (2 * np.pi * np.abs(np.linalg.det(cov[k].reshape(2, 2)))) ** -0.5
            gamma[:,k] = pi[k] * exp*con
        gamma = gamma / np.sum(gamma,axis=1)[:,None]
        for k in range(K):
            mu[k] = np.sum(np.multiply(gamma[:, k][:,None],X), axis = 0)/np.sum(gamma[:,k],axis=0)
            cov[k] = (np.dot(np.multiply(X - mu[k], gamma[:, k][:,None]).T, X - mu[k])/np.sum(gamma[:,k],axis=0)).reshape(4)
            pi[k] = np.sum(gamma[:,k],axis=0) / X.shape[0]
    return mu.tolist(), cov.tolist()

def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()
