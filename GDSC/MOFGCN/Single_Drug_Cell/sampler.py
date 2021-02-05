import torch
import numpy as np


class Sampler(object):
    """
    对指定目标进行采样
    采样返回结果为torch.tensor
    """
    def __init__(self, origin_adj_mat, null_mask, target_index, train_index, test_index):
        super(Sampler, self).__init__()
        self.adj_mat = origin_adj_mat
        self.null_mask = null_mask
        self.target_index = target_index
        self.train_index = train_index
        self.test_index = test_index
        self.train_data, self.test_data = self.sample_train_test_data()
        self.train_mask, self.test_mask = self.sample_train_test_mask()

    def sample_train_test_data(self):
        test_data = np.zeros(self.adj_mat.shape, dtype=np.float32)
        test_data[self.test_index, self.target_index] = 1
        train_data = self.adj_mat - test_data
        train_data = torch.from_numpy(train_data)
        test_data = torch.from_numpy(test_data)
        return train_data, test_data

    def sample_train_test_mask(self):
        neg_value = np.ones(self.adj_mat.shape, dtype=np.float32)
        neg_value = neg_value - self.adj_mat - self.null_mask
        neg_test_mask = np.zeros(self.adj_mat.shape, dtype=np.float32)
        target_neg_index = np.where(neg_value[:, self.target_index] == 1)[0]
        target_neg_test_index = np.random.choice(target_neg_index, self.test_index.shape[0], replace=False)
        neg_test_mask[target_neg_test_index, self.target_index] = 1
        neg_value[target_neg_test_index, self.target_index] = 0
        train_mask = (self.train_data.numpy() + neg_value).astype(np.bool)
        test_mask = (self.test_data.numpy() + neg_test_mask).astype(np.bool)
        train_mask = torch.from_numpy(train_mask)
        test_mask = torch.from_numpy(test_mask)
        return train_mask, test_mask
