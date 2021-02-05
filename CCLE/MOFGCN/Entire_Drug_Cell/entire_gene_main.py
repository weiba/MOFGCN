# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
from import_path import *
from MOFGCN.model import GModel
from MOFGCN.optimizer import Optimizer
from sklearn.model_selection import KFold
from MOFGCN.Entire_Drug_Cell.sampler import Sampler
from MOFGCN.myutils import roc_auc, translate_result


data_dir = dir_path(k=2) + "processed_data/"

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
adj_mat_coo_data = sp.coo_matrix(cell_drug).data

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)

# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)

# 加载细胞系-cna特征矩阵
cna = np.eye(gene.shape[0], dtype=np.float32)

# 加载细胞系-mutaion特征矩阵
mutation = np.eye(gene.shape[0], dtype=np.float32)

# 加载null_mask
null_mask = np.zeros(cell_drug.shape, dtype=np.float32)

epochs = []
true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=11)

n_kfold = 5
for fold in range(n_kfold):
    for train_index, test_index in kfold.split(np.arange(adj_mat_coo_data.shape[0])):
        sampler = Sampler(cell_drug, train_index, test_index, null_mask)
        model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, sigma=2, k=2, iterates=3,
                       feature_drug=feature_drug, n_hid1=192, n_hid2=64, alpha=8.70, device="cuda:0")
        opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                        roc_auc, lr=1e-3, epochs=1000, device="cuda:0").to("cuda:0")
        epoch, true_data, predict_data = opt()
        epochs.append(epoch)
        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(predict_data))
file = open("./result_data/epochs_gene.txt", "w")
file.write(str(epochs))
file.close()
pd.DataFrame(true_datas).to_csv("./result_data/true_data_gene.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data_gene.csv")
