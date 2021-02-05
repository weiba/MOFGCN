import torch
from abc import ABC
import torch.nn as nn
import torch.optim as optim
from MOFGCN.model import EarlyStop
from MOFGCN.myutils import cross_entropy_loss


class Optimizer(nn.Module, ABC):
    def __init__(self, model, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.01, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer, self).__init__()
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device)
        self.train_mask = train_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)
        early_stop = EarlyStop(tolerance=8, data_len=true_data.size()[0])
        for epoch in torch.arange(self.epochs):
            predict_data = self.model()
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % self.test_freq == 0:
                predict_data_masked = torch.masked_select(predict_data, self.test_mask)
                auc = self.evaluate_fun(true_data, predict_data_masked)
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % auc)
                flag = early_stop.stop(auc=auc, epoch=epoch.item(), predict_data=predict_data_masked)
                if flag:
                    break
        print("Fit finished.")
        max_index = early_stop.get_best_index()
        best_epoch = early_stop.epoch_pre[max_index]
        best_predict = early_stop.predict_data_pre[max_index, :]
        return best_epoch, true_data, best_predict
