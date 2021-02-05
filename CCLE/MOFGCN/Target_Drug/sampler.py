import torch
import numpy as np
import scipy.sparse as sp


class Sampler(object):
    def __init__(self, response_mat: np.ndarray, null_mask: np.ndarray, target_indexes: np.ndarray,
                 pos_train_index: np.ndarray, pos_test_index: np.ndarray):
        self.response_mat = response_mat
        self.null_mask = null_mask
        self.target_indexes = target_indexes
        self.pos_train_index = pos_train_index
        self.pos_test_index = pos_test_index
        self.train_data, self.test_data = self.sample_train_test_data()
        self.train_mask, self.test_mask = self.sample_train_test_mask()

    def sample_train_test_data(self):
        n_target = self.target_indexes.shape[0]
        target_response = self.response_mat[:, self.target_indexes].reshape((-1, n_target))
        train_data = self.response_mat.copy()
        train_data[:, self.target_indexes] = 0
        target_pos_value = sp.coo_matrix(target_response)
        target_train_data = sp.coo_matrix((target_pos_value.data[self.pos_train_index],
                                           (target_pos_value.row[self.pos_train_index],
                                            target_pos_value.col[self.pos_train_index])),
                                          shape=target_response.shape).toarray()
        target_test_data = sp.coo_matrix((target_pos_value.data[self.pos_test_index],
                                          (target_pos_value.row[self.pos_test_index],
                                           target_pos_value.col[self.pos_test_index])),
                                         shape=target_response.shape).toarray()
        test_data = np.zeros(self.response_mat.shape, dtype=np.float32)
        for i, value in enumerate(self.target_indexes):
            train_data[:, value] = target_train_data[:, i]
            test_data[:, value] = target_test_data[:, i]
        train_data = torch.from_numpy(train_data)
        test_data = torch.from_numpy(test_data)
        return train_data, test_data

    def sample_train_test_mask(self):
        target_response = self.response_mat[:, self.target_indexes]
        target_ones = np.ones(target_response.shape, dtype=np.float32)
        target_neg_value = target_ones - target_response - self.null_mask[:, self.target_indexes]
        target_neg_value = sp.coo_matrix(target_neg_value)
        ids = np.arange(target_neg_value.data.shape[0])
        target_neg_test_index = np.random.choice(ids, self.pos_test_index.shape[0], replace=False)
        target_neg_test_mask = sp.coo_matrix((target_neg_value.data[target_neg_test_index],
                                              (target_neg_value.row[target_neg_test_index],
                                               target_neg_value.col[target_neg_test_index])),
                                             shape=target_response.shape).toarray()
        neg_test_mask = np.zeros(self.response_mat.shape, dtype=np.float32)
        for i, value in enumerate(self.target_indexes):
            neg_test_mask[:, value] = target_neg_test_mask[:, i]
        other_neg_value = np.ones(self.response_mat.shape,
                                  dtype=np.float32) - neg_test_mask - self.response_mat - self.null_mask
        test_mask = (self.test_data.numpy() + neg_test_mask).astype(np.bool)
        train_mask = (self.train_data.numpy() + other_neg_value).astype(np.bool)
        test_mask = torch.from_numpy(test_mask)
        train_mask = torch.from_numpy(train_mask)
        return train_mask, test_mask
