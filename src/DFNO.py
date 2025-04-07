# Detedting fuzzy neighborhood outliers (DFNO) algorithm
# Please refer to the following papers:
# Yuan Zhong,Hu Peng, Chen Hongmei, Chen Yingke, and Li Qilin.Detecting fuzzy neighoborhood outliers[J].
# IEEE Transactions on Knowledge and Data Engineering,2024.
# Uploaded by Yuan Zhong on October 30, 2024. E-mail: yuanzhong@scu.edu.cn or yuanzhong2799@foxmail.com.
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler

def DFNO(data, affinity_matrix , k , is_similarity_matrix=False):
    # input:
    # data is data matrix without decisions, where rows for samples and columns for attributes.
    # Numerical attributes should be normalized into [0,1].
    # Nominal attributes be replaced by different integer values.
    # k: int, parameter for k-fuzzy nearest neighbor
    # output
    # FNOS: numpy array, Fuzzy-neighborhood outlier score

    n = data.shape[0]

    # Compute fuzzy similarity relation
    if is_similarity_matrix:
        sim = affinity_matrix
    else:
        sim = FSR(data)
    #sim = np.multiply(FSR(data),affinity_matrix)
    #sim = affinity_matrix
    # Sort similarity matrix and find k-th similarity
    similarity = np.sort(sim, axis=1)[:, ::-1]
    num = np.argsort(-sim, axis=1, kind='stable')  # Indices of sorted similarity in descending order
    ksimilarity = similarity[:, k]
    fkNN_temp = np.where(sim >= ksimilarity[:, None], sim, 0)

    fkNN_card = np.sum(fkNN_temp, axis=1)
    count = np.sum(fkNN_temp != 0, axis=1)

    # Calculate reachability similarity
    reachsim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            reachsim[i, j] = min(sim[i, j], ksimilarity[j])

    # Local reachability density (lrd) calculation
    lrd = np.zeros(n)
    for i in range(n):
        sum_reachdist = 0
        for j in range(count[i]):
            sum_reachdist += reachsim[i, num[i, j + 1]]
        if fkNN_card[i] > 0 :
            lrd[i] = sum_reachdist / fkNN_card[i]
        else:
            lrd[i] = 1e-11

    # Fuzzy local density deviation (FLDD) calculation
    FLDD = np.zeros(n)
    for i in range(n):
        sumlrd = 0
        for j in range(count[i]):
            sumlrd += lrd[num[i, j + 1]] / lrd[i]
        if lrd[i] > 0 and fkNN_card[i] > 0:
            FLDD[i] = sumlrd / fkNN_card[i]
        else:
            FLDD[i] = 1.0

    FNOS = FLDD
    return FNOS

def FSR(data):
    # Fuzzy Similarity Relation
    n, m = data.shape

    # Numerical and nominal feature separation
    num_fea = np.all(data <= 1, axis=0)
    nom_fea = ~num_fea

    num_dis = squareform(pdist(data[:, num_fea])) if num_fea.any() else np.zeros((n, n))
    nom_dis = squareform(pdist(data[:, nom_fea], metric='hamming')) if nom_fea.any() else np.zeros((n, n))

    dis = num_dis + nom_dis * np.sum(nom_fea)
    fsr = 1 - dis / m
    return fsr

if __name__ == "__main__":
    load_data = loadmat('Example.mat')
    trandata = load_data['Example']
    scaler = MinMaxScaler()
    trandata[:, 0:2] = scaler.fit_transform(trandata[:, 0:2])

    k = 3
    anomaly_scores = DFNO(trandata, k)
    print(anomaly_scores)
