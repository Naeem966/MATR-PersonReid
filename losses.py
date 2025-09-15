import torch
import torch.nn as nn
import torch.nn.functional as F

class PartAwareTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        B, P, D = features.size()
        features = features.view(B*P, D)
        labels = labels.repeat_interleave(P)
        loss = nn.TripletMarginLoss(margin=self.margin)(features, features, features)
        return loss

class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.rce = lambda pred, target: -torch.sum(F.one_hot(target, pred.size(1)) * F.log_softmax(pred, 1), 1).mean()
    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + self.beta * self.rce(pred, target)

def mutual_learning_loss(outputs_list, targets_list):
    loss = 0
    for i in range(len(outputs_list)):
        for j in range(i + 1, len(outputs_list)):
            loss += F.mse_loss(F.softmax(outputs_list[i], 1), F.softmax(outputs_list[j], 1))
    return loss