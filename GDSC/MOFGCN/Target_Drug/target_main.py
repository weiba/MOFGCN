# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
from import_path import *
from MOFGCN.model import GModel
from MOFGCN.optimizer import Optimizer
from sklearn.model_selection import KFold
from MOFGCN.Target_Drug.sampler import Sampler
from MOFGCN.myutils import roc_auc, translate_result, common_data_index


data_dir = dir_path(k=2) + "processed_data/"
target_drug_cids = np.array([11626560, 16038120, 49806720, 73265211])

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_common_binary.csv", index_col=0, header=0)
cell_drug.columns = cell_drug.columns.astype(np.int)
drug_cids = cell_drug.columns.values
cell_target_drug = np.array(cell_drug.loc[:, target_drug_cids], dtype=np.float32)
adj_mat_coo_data = sp.coo_matrix(cell_target_drug).data
cell_drug = np.array(cell_drug, dtype=np.float32)

target_indexes = common_data_index(drug_cids, target_drug_cids)

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)

# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "cell_gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)

# 加载细胞系-cna特征矩阵
cna = pd.read_csv(data_dir + "cell_gene_cna.csv", index_col=0, header=0)
cna = cna.fillna(0)
cna = np.array(cna, dtype=np.float32)

# 加载细胞系-mutaion特征矩阵
mutation = pd.read_csv(data_dir + "cell_gene_mutation.csv", index_col=0, header=0)
mutation = np.array(mutation, dtype=np.float32)

# 加载null_mask
null_mask = pd.read_csv(data_dir + "null_mask.csv", index_col=0, header=0)
null_mask = np.array(null_mask, dtype=np.float32)

epochs = []
true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=21)

n_kfolds = 5
for fold in range(n_kfolds):
    for train_index, test_index in kfold.split(np.arange(adj_mat_coo_data.shape[0])):
        sampler = Sampler(response_mat=cell_drug, null_mask=null_mask, target_indexes=target_indexes,
                          pos_train_index=train_index, pos_test_index=test_index)
        model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, sigma=2, k=11, iterates=3,
                       feature_drug=feature_drug, n_hid1=192, n_hid2=36, alpha=5.74, device="cuda:0")
        opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                        roc_auc, lr=5e-4, epochs=1000, device="cuda:0").to("cuda:0")
        epoch, true_data, predict_data = opt()
        epochs.append(epoch)
        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(predict_data))
file = open("./result_data/epochs.txt", "w")
file.write(str(epochs))
file.close()
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")
