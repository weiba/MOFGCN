import torch
import numpy as np
from abc import ABC
import torch.nn as nn
import torch.nn.functional as fun
from MOFGCN.myutils import exp_similarity, full_kernel, sparse_kernel, jaccard_coef, torch_corr_x_y, \
    scale_sigmoid_activation_function


class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="cpu"):
        super(ConstructAdjMatrix, self).__init__()
        self.adj_mat = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        n_cell = self.adj_mat.shape[0]
        n_drug = self.adj_mat.shape[1]
        cell_identity = torch.diag(torch.diag(torch.ones(n_cell, n_cell, dtype=torch.float, device=self.device)))
        drug_identity = torch.diag(torch.diag(torch.ones(n_drug, n_drug, dtype=torch.float, device=self.device)))
        cell_drug = torch.cat((cell_identity, self.adj_mat), dim=1)
        drug_cell = torch.cat((torch.t(self.adj_mat), drug_identity), dim=1)
        adj_matrix = torch.cat((cell_drug, drug_cell), dim=0)
        d = torch.diag(torch.pow(torch.sum(adj_matrix, dim=1), -1/2))
        identity = torch.diag(torch.diag(torch.ones(d.shape, dtype=torch.float, device=self.device)))
        adj_matrix_hat = torch.add(identity, torch.mm(d, torch.mm(adj_matrix, d)))
        return adj_matrix_hat


class FusionFeature(nn.Module, ABC):
    def __init__(self, gene, cna, mutation, sigma, k, iterates, feature_drug, device="cpu"):
        super(FusionFeature, self).__init__()
        gene = torch.from_numpy(gene).to(device)
        cna = torch.from_numpy(cna).to(device)
        mutation = torch.from_numpy(mutation).to(device)
        sigma = torch.tensor(sigma, dtype=torch.float, device=device)
        feature_drug = torch.from_numpy(feature_drug).to(device)
        self.gene_exp_similarity = exp_similarity(gene, sigma)
        self.cna_exp_similarity = exp_similarity(cna, sigma, normalize=False)
        self.mutation_exp_similarity = exp_similarity(mutation, sigma, normalize=False)
        self.drug_jac_similarity = jaccard_coef(feature_drug)
        self.k = k
        self.iterates = iterates
        self.device = device

    def fusion_cell_feature(self):
        gene_p = full_kernel(self.gene_exp_similarity)
        gene_s = sparse_kernel(self.gene_exp_similarity, k=self.k)
        cna_p = full_kernel(self.cna_exp_similarity)
        cna_s = sparse_kernel(self.cna_exp_similarity, k=self.k)
        mutation_p = full_kernel(self.mutation_exp_similarity)
        mutation_s = sparse_kernel(self.mutation_exp_similarity, k=self.k)
        two = torch.tensor(2, dtype=torch.float32, device=self.device)
        three = torch.tensor(3, dtype=torch.float32, device=self.device)
        it = 0
        while it < self.iterates:
            gene_p_next = torch.mm(torch.mm(gene_s, torch.div(torch.add(cna_p, mutation_p), two)), gene_s.t())
            cna_p_next = torch.mm(torch.mm(cna_s, torch.div(torch.add(gene_p, mutation_p), two)), cna_s.t())
            mutation_p_next = torch.mm(torch.mm(mutation_s, torch.div(torch.add(cna_p, gene_p), two)), mutation_s.t())
            gene_p = gene_p_next
            cna_p = cna_p_next
            mutation_p = mutation_p_next
            it += 1
        fusion_feature = torch.div(torch.add(torch.add(gene_p, cna_p), mutation_p), three)
        fusion_feature = fusion_feature.to(dtype=torch.float32)
        return fusion_feature

    def forward(self):
        drug_similarity = full_kernel(self.drug_jac_similarity)
        cell_similarity = self.fusion_cell_feature()
        zeros1 = torch.zeros(cell_similarity.shape[0], drug_similarity.shape[1], dtype=torch.float32,
                             device=self.device)
        zeros2 = torch.zeros(drug_similarity.shape[0], cell_similarity.shape[1], dtype=torch.float32,
                             device=self.device)
        cell_zeros = torch.cat((cell_similarity, zeros1), dim=1)
        zeros_drug = torch.cat((zeros2, drug_similarity), dim=1)
        fusion_feature = torch.cat((cell_zeros, zeros_drug), dim=0)
        return fusion_feature


class GEncoder(nn.Module, ABC):
    def __init__(self, adj_mat, feature, n_hid):
        super(GEncoder, self).__init__()
        self.adj_mat = adj_mat
        self.feature = feature
        self.lm = nn.Linear(feature.shape[1], n_hid, bias=False)

    def forward(self):
        input_adj_mat = torch.mm(self.adj_mat, self.feature)
        lm_out = self.lm(input_adj_mat)
        lm_out = fun.relu(lm_out)
        return lm_out


class GDecoder(nn.Module, ABC):
    def __init__(self, n_cell, n_drug, n_hid1, n_hid2, alpha):
        super(GDecoder, self).__init__()
        self.n_cell = n_cell
        self.n_drug = n_drug
        self.alpha = alpha
        self.lm_cell = nn.Linear(n_hid1, n_hid2, bias=False)
        self.lm_drug = nn.Linear(n_hid1, n_hid2, bias=False)

    def forward(self, encode_output):
        z_cell, z_drug = torch.split(encode_output, [self.n_cell, self.n_drug], dim=0)
        cell = self.lm_cell(z_cell)
        drug = self.lm_drug(z_drug)
        output = torch_corr_x_y(cell, drug)
        output = scale_sigmoid_activation_function(output, alpha=self.alpha)
        return output


class GModel(nn.Module, ABC):
    def __init__(self, adj_mat, gene, cna, mutation, sigma, k, iterates, feature_drug, n_hid1, n_hid2, alpha,
                 device="cpu"):
        super(GModel, self).__init__()
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        fusioner = FusionFeature(gene, cna, mutation, sigma=sigma, k=k, iterates=iterates, feature_drug=feature_drug,
                                 device=device)
        adj_matrix_hat = construct_adj_matrix()
        feature = fusioner()
        self.encoder = GEncoder(adj_matrix_hat, feature, n_hid1)
        self.decoder = GDecoder(adj_mat.shape[0], adj_mat.shape[1], n_hid1=n_hid1, n_hid2=n_hid2, alpha=alpha)

    def forward(self):
        encode_output = self.encoder()
        output = self.decoder(encode_output)
        return output


class Early(object):
    def __init__(self, tolerance: int, data_len: int):
        self.auc = np.zeros(tolerance, dtype=np.float32)
        self.epoch = np.zeros(tolerance, dtype=np.int)
        self.predict_data = torch.zeros((tolerance, data_len), dtype=torch.float32)
        self.tolerance = tolerance
        self.len = 0

    def push_data(self, auc, epoch, predict_data):
        i = self.len % self.tolerance
        self.auc[i] = auc
        self.epoch[i] = epoch
        self.predict_data[i, :] = predict_data
        self.len += 1

    def average(self):
        if self.len < self.tolerance:
            avg = 0
        else:
            avg = np.mean(self.auc)
        return avg


class EarlyStop(object):
    def __init__(self, tolerance: int, data_len: int):
        self.early = Early(tolerance=tolerance, data_len=data_len)
        self.auc_pre = None
        self.epoch_pre = None
        self.predict_data_pre = None

    def stop(self, auc, epoch, predict_data):
        avg_pre = self.early.average()
        self.auc_pre = self.early.auc.copy()
        self.epoch_pre = self.early.epoch.copy()
        self.predict_data_pre = self.early.predict_data.clone()
        self.early.push_data(auc=auc, epoch=epoch, predict_data=predict_data)
        avg_next = self.early.average()
        flag = False
        if avg_pre > avg_next:
            flag = True
        return flag

    def get_best_index(self):
        best_index = np.argmax(self.auc_pre)
        if self.epoch_pre[best_index] == 0:
            self.auc_pre[best_index] = 0
            best_index = np.argmax(self.auc_pre)
        return best_index
