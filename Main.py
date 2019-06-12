import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from LinkPrediction import LinkPredict
from LinkPrediction import Evaluation
from tkinter import filedialog
import tkinter

# GUI 打开数据集
root = tkinter.Tk()
root.geometry('200x100')
filename = filedialog.askopenfilename(title='打开数据集文件', filetype=[('All Files', '*'), ])
root.mainloop()
root.quit()

str = input("读取的数据集是：")
#加载数据集,分为3个函数：1.是否需要跳过，跳过几行 2.是否有权重
#i：跳过几行
def needSkip(i):
    Data = np.loadtxt(filename, skiprows=i)
    return Data

if str=='jazz':
    i=3
    flag=True
elif (str=='yeast')or(str=='politicalBlogs')or(str=='router'):
    i=0
    flag=False
elif (str=='hamster'):
    i=1
    flag=False
elif (str =='foodweb')or(str=='foodweb2'):
    i=2
    flag=True
elif str =='contact':
    i=1
    flag=True
elif str =='worldTrade':
    i=83
    flag=True
elif str=='celegans':
    i=456
    flag=True
elif str == 'usair':
    i = 0
    flag = True
Data = needSkip(i)

#邻接矩阵的规模
def findNodes(DataList):
    list_a = []
    list_b = []
    for row in range(DataList.shape[0]):
        list_a.append(DataList[row][0])
        list_b.append(DataList[row][1])

    list_a = list(set(list_a))
    list_b = list(set(list_b))
    len_a = int(max(list_a))
    len_b = int(max(list_b))
    nodes = max(len_a, len_b) + 1  # 确定最大的节点数+1(便于构造数组)

    print('Edges=', DataList.shape[0])
    print('Nodes=', nodes - 1)
    return nodes
nodes = findNodes(Data)

def createMatrixAdjacency (DataList,flag):
    '''
    构造邻接矩阵
    :param DataList:
    :return: MatrixAdjacency
    '''
    MatrixAdjacency = np.zeros([nodes,nodes])
    for col in range(DataList.shape[0]):
        i = int(DataList[col][0])
        j = int(DataList[col][1])
        if flag:
            MatrixAdjacency[i][j] = 1
            MatrixAdjacency[j][i] = 1
        else:
            MatrixAdjacency[i][j] = 1
            MatrixAdjacency[j][i] = 1
    return MatrixAdjacency



print(filename)
# 划分训练集，测试集
x_train ,x_test = model_selection.train_test_split(Data)



matrix_train = createMatrixAdjacency(x_train,flag)
matrix_test = createMatrixAdjacency(x_test,flag)

#precision指标
print('Precision')
pre_CN = Evaluation.Precision(matrix_train,matrix_test,LinkPredict.CN(matrix_train),nodes)
pre_Ja = Evaluation.Precision(matrix_train,matrix_test,LinkPredict.Jaccard(matrix_train),nodes)
pre_RA = Evaluation.Precision(matrix_train,matrix_test,LinkPredict.RA(matrix_train),nodes)
pre_AA = Evaluation.Precision(matrix_train,matrix_test,LinkPredict.AA(matrix_train),nodes)
pre_PA = Evaluation.Precision(matrix_train,matrix_test,LinkPredict.PA(matrix_train),nodes)
pre_Katz = Evaluation.Precision(matrix_train,matrix_test,LinkPredict.Katz(matrix_train),nodes)
pre_CRA = Evaluation.Precision(matrix_train,matrix_test,LinkPredict.CRA(matrix_train),nodes)


precision_y=[pre_CN,pre_Ja,pre_RA,pre_AA,pre_PA,pre_Katz,pre_CRA]
# 设置y轴数据，以数组形式提供
precision_x=['CN','Jaccard','RA','AA','PA','Katz','CRA']    # 以0开始的递增序列作为x轴数据

#AUC指标
print("AUC")
auc_CN = Evaluation.AUC(matrix_train,matrix_test,LinkPredict.CN(matrix_train),nodes)
auc_Ja = Evaluation.AUC(matrix_train,matrix_test,LinkPredict.Jaccard(matrix_train),nodes)
auc_RA =  Evaluation.AUC(matrix_train,matrix_test,LinkPredict.RA(matrix_train),nodes)
auc_AA =  Evaluation.AUC(matrix_train,matrix_test,LinkPredict.AA(matrix_train),nodes)
auc_PA =  Evaluation.AUC(matrix_train,matrix_test,LinkPredict.PA(matrix_train),nodes)
auc_Katz =  Evaluation.AUC(matrix_train,matrix_test,LinkPredict.Katz(matrix_train),nodes)
auc_CRA =  Evaluation.AUC(matrix_train,matrix_test,LinkPredict.CRA(matrix_train),nodes)


auc_y=[auc_CN,auc_Ja,auc_RA,auc_AA,auc_PA,auc_Katz,auc_CRA] #
auc_x=['CN','Jaccard','RA','AA','PA','Katz','CRA']    # 以0开始的递增序列作为x轴数据


# plot画图
plt.figure(figsize=(8,8),dpi=80)

plt.subplot(211)
plt.title('Precision')
plt.plot(precision_x,precision_y) # 在2x1画布中第一块区域画出Precison指标
plt.subplot(212)
plt.title('AUC')
plt.plot(auc_x,auc_y) # 在2x2画布中第二块区域画出AUC指标

plt.show()