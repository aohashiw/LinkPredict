import numpy as np
import math

def CN(matrix_train):
    '''
    CommonNeighbors算法： score[i,j] = 节点i，j的共同邻居数
    :param matrix_train:训练集邻接矩阵
    :return:matrix_score:评分矩阵
    '''
    # 矩阵相乘 matrix_score[i,j]即i，与j的共同邻居数
    matrix_score = np.dot(matrix_train,matrix_train)
    return matrix_score

def Jaccard(matrix_train):
    '''
    Jaccard 算法: score[i,j]=CN_ij/(degree_i+degree_j-CN_ij)

    :param matrix_train: 训练集邻接矩阵
    :return:matrix_score: 评分矩阵
    '''
    matrix_score = np.dot(matrix_train,matrix_train) #得到CN矩阵
    nodeNums = matrix_train.shape[0]
    degree = sum(matrix_train) #计算每个节点的度
    degree.shape = (degree.shape[0],1)
    degree_T = degree.T
    degree_sum = degree+degree_T #生成degree矩阵 degree_sum[i,j]=degree(i)+degree(j)
    temp = degree_sum-matrix_score # temp[i,j]=degree(i)+degree(j)-len(CN(i,j))
    for i in range(1,nodeNums):
        for j in range(1,nodeNums):
            if temp[i][j]!= 0:
                # matrix_score[i,j]=len(CN(i,j))/{degree(i)+degree(j)-len(CN(i,j))}
                matrix_score[i][j]=matrix_score[i][j]/temp[i][j]
    return matrix_score

def RA(matrix_train):
    '''
    RA算法: score[i,j] = 求和(1/degree_z) ，其中z属于i，j的Common neighbor节点
    :param matrix_train: 训练集邻接矩阵
    :return:matrix_score: 评分矩阵
    '''
    degree_row = sum(matrix_train) #计算每个节点的度
    degree_row.shape = (degree_row.shape[0],1)
    matrix_temp = matrix_train/degree_row
    matrix_temp = np.nan_to_num(matrix_temp) # 将nan值转化为0
    matrix_score = np.dot(matrix_train,matrix_temp)
    return matrix_score

def AA(matrix_train):
    '''
    AA算法: score[i,j] = 求和(1/log(degree_z)) ，其中z属于i，j的Common neighbor节点
    :param matrix_train: 训练集邻接矩阵
    :return:matrix_score: 评分矩阵
    '''
    log_degree_row = np.log(sum(matrix_train)) #计算每个节点的度并取对数
    log_degree_row = np.nan_to_num(log_degree_row)
    log_degree_row.shape = (log_degree_row.shape[0],1)
    matrix_temp = matrix_train / log_degree_row
    matrix_temp = np.nan_to_num(matrix_temp)
    matrix_score = np.dot(matrix_train, matrix_temp)
    return matrix_score

def PA(matrix_train):
    '''
    PA算法: Sxy = Kx*Ky
    :param matrix_train:
    :return: matrix_score
    '''
    degree_row = sum(matrix_train)
    degree_row.shape = (degree_row.shape[0],1)
    degree_row_T = degree_row.T
    matrix_score = np.dot(degree_row,degree_row_T)
    return matrix_score



def Katz(matrix_train):
    '''
    Katz算法： Score = beta*A+beta^2*A^2+beta^3*A^3·····=（I-beta*A）^(-1)-I
    :param matrix_train:
    :return:
    '''
    Beta = 0.01
    nodeNums = matrix_train.shape[0]
    e, v = np.linalg.eig(matrix_train)
    value = max(e)
    while Beta * value >1:
        # 确保收敛
        Beta = Beta/2


    matrix_I = np.eye(nodeNums)#生成单位矩阵I
    matrix_temp = matrix_I-Beta*matrix_train# temp=I-beta*A
    matrix_temp = np.linalg.inv(matrix_temp)# 求逆
    matrix_score = matrix_temp-matrix_I

    return matrix_score

def CRA(matrix_train):
    '''
    RA in community
    :param matrix_train:
    :return:
    '''
    nodeNums = matrix_train.shape[0]
    Adj = [set(),]
    for i in range(1, nodeNums):
        adj = matrix_train[i]
        b = adj.nonzero()
        Adj.insert(i, set(b[0].tolist()))

    matrix_score = np.zeros([nodeNums, nodeNums])

    for i in range(1, nodeNums):
        for j in range(1, nodeNums):
            cn = Adj[i].intersection(Adj[j])
            score = 0
            for x in cn:
                cnn = cn.intersection(Adj[x])
                if len(Adj[x])>0:
                    score = score+ len(cnn)/len(Adj[x])
            matrix_score[i][j]=score

    return matrix_score