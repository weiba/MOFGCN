# coding: utf-8
import time
import numpy as np
import pandas as pd
from import_path import *
from sklearn.model_selection import KFold
from MOFGCN.myutils import translate_result, roc_auc
from MOFGCN.Single_Drug_Cell.MOFGCN_Single_target import mofgcn_single_target


data_dir = dir_path(k=2) + "processed_data/"

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
cell_sum = np.sum(cell_drug, axis=1)
drug_sum = np.sum(cell_drug, axis=0)

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)

# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)

# 加载细胞系-cna特征矩阵
cna = pd.read_csv(data_dir + "cna_feature.csv", index_col=0, header=0)
cna = np.array(cna, dtype=np.float32)

# 加载细胞系-mutaion特征矩阵
mutation = pd.read_csv(data_dir + "mutation_feature.csv", index_col=0, header=0)
mutation = np.array(mutation, dtype=np.float32)

# 加载null_mask
null_mask = np.zeros(cell_drug.shape, dtype=np.float32)

k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=21)

file_drug = open("./pan_result_data/single_drug_epoch.txt", "w")

file_drug_time = open("./pan_result_data/each_drug_time.txt", "w")

n_kfolds = 5
for target_index in np.arange(cell_drug.shape[1]):
    times = []
    epochs = []
    true_data_s = pd.DataFrame()
    predict_data_s = pd.DataFrame()
    target_pos_index = np.where(cell_drug[:, target_index] == 1)[0]
    if drug_sum[target_index] < 10:
        continue
    for folds in range(n_kfolds):
        start = time.time()
        for train, test in kfold.split(target_pos_index):
            train_index = target_pos_index[train]
            test_index = target_pos_index[test]
            epoch, true_data, predict_data = mofgcn_single_target(gene=gene, cna=cna, mutation=mutation,
                                                                  drug_feature=feature_drug, response_mat=cell_drug,
                                                                  null_mask=null_mask, target_index=target_index,
                                                                  train_index=train_index, test_index=test_index,
                                                                  evaluate_fun=roc_auc, device="cuda:0")
            epochs.append(epoch)
            true_data_s = true_data_s.append(translate_result(true_data))
            predict_data_s = predict_data_s.append(translate_result(predict_data))
        end = time.time()
        times.append(end - start)
    file_drug.write(str(target_index) + ":" + str(epochs) + "\n")
    file_drug_time.write(str(target_index) + ":" + str(times) + "\n")
    true_data_s.to_csv("./pan_result_data/drug_" + str(target_index) + "_" + "true_data.csv")
    predict_data_s.to_csv("./pan_result_data/drug_" + str(target_index) + "_" + "predict_data.csv")
file_drug.close()
file_drug_time.close()
