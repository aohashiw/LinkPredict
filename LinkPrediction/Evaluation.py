import numpy as np



def AUC(matrix_train,matrix_test,matrix_score,MaxNodeNum):
    '''
    AUC 指标： AUC = n'+0.5n''/AUCnum
    :param matrix_train: 训练集邻接矩阵
    :param matrix_test: 测试集邻接矩阵
    :param matrix_score: Score矩阵
    :param MaxNodeNum: 节点数+1
    :return: auc AUC指标
    '''
    AUCnum = MaxNodeNum*3 # 随机选取的边的数目
    matrix_score = np.triu(matrix_score) #取上三角矩阵
    matrix_NoExist = np.ones(MaxNodeNum) - matrix_train - matrix_test - np.eye(MaxNodeNum) # 求待预测集的邻接矩阵

    Test = np.triu(matrix_test) #取上三角矩阵
    NoExist = np.triu(matrix_NoExist) #取上三角矩阵

    Test_num = len(np.argwhere(Test>0))  #测试集中的边数
    NoExist_num = len(np.argwhere(NoExist>0)) # 带预测的边数

    Test_rd = [int(x) for index, x in enumerate((Test_num * np.random.rand(1, AUCnum))[0])] #随机生成 AUCnum条测试集中的边
    NoExist_rd = [int(x) for index, x in enumerate((NoExist_num * np.random.rand(1, AUCnum))[0])] #随机生成 AUCnum条待预测集中的边

    TestPre = matrix_score * Test #测试集中每条边的分数Score矩阵，其余分数置0
    NoExistPre = matrix_score * NoExist # 待预测集中每条边的分数Score矩阵，其余分数置0

    TestIndex = np.argwhere(Test > 0) # 测试集中的边的节点索引i，j
    Test_Data = np.array([TestPre[x[0], x[1]] for index, x in enumerate(TestIndex)]).T # 以TestIndex为基准获取对应边的Score

    NoExistIndex = np.argwhere(NoExist >0) # 带预测集中的边的节点索引i，j
    NoExist_Data = np.array([NoExistPre[x[0], x[1]] for index, x in enumerate(NoExistIndex)]).T # 以NoExistIndex为基准获取对应边的Score

    Test_rd = np.array([Test_Data[x] for index, x in enumerate(Test_rd)]) # 将随机生成的边转换为其Score
    NoExist_rd = np.array([NoExist_Data[x] for index, x in enumerate(NoExist_rd)])

    n1, n2 = 0, 0
    for num in range(AUCnum):
        if Test_rd[num] > NoExist_rd[num]:
            n1 += 1
        elif Test_rd[num] == NoExist_rd[num]:
            n2 += 0.5
        else:
            n1 += 0
    auc = float(n1 + n2) / AUCnum
    # AUC = n'+0.5n''/AUCnum
    print(auc)
    return auc



def Precision(matrix_train,matrix_test,matrix_score,MaxNodeNum):
    '''
    Precision指标：m/L
    :param matrix_train:
    :param matrix_test:
    :param matrix_score:
    :param MaxNodeNum:
    :return:
    '''
    L = MaxNodeNum*3
    predict=[]
    for i in range(1,MaxNodeNum):
        for j in range(i,MaxNodeNum):
            if(i!=j and matrix_train[i][j]==0):
                # 如果节点i与j在训练集中无连边，就在预测集中追加该数据
                predict.append((i,j,matrix_score[i][j]))

    dtype = [('Node1', int), ('Node2', int), ('Score', float)]
    nm = np.array(predict, dtype=dtype)
    nm = np.sort(nm, order=['Score', ]) #按照Score对测试集排序

    # 选取预测集中评分最高的L条数据
    new_nm = nm[nm.shape[0] - L:nm.shape[0]]
    m=0
    for x in new_nm:
        # x = (node1,node2,score), 若预测的边（node1,node2)存在于测试集中，分数+1
        if matrix_test[x[0]][x[1]]>0:
            m=m+1
    precision = m/L
    print(precision)

    return precision