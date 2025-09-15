import numpy as np
from sklearn.cluster import DBSCAN
import torch

def extract_features(model, loader, device):
    model.eval()
    feats, paths = [], []
    for batch in loader:
        img = batch['image'].to(device)
        attr = batch['attr_emb'].to(device)
        with torch.no_grad():
            f = model(img, attr, return_features=True).cpu().numpy()
        feats.append(f)
        paths.extend(batch['path'])
    feats = np.concatenate(feats, 0)
    return feats, paths

def run_dbscan(feats, eps=0.6, min_samples=4):
    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(feats)
    return cluster.labels_

def consensus_topk_neighbors(feats_list, k=5):
    consensus = []
    for i in range(len(feats_list[0])):
        sets = [set(np.argsort(np.linalg.norm(f[i] - f, axis=1))[:k]) for f in feats_list]
        consensus.append(set.intersection(*sets))
    return consensus

def compute_confidence(feats, centroids, labels):
    conf = []
    for i, f in enumerate(feats):
        c = centroids[labels[i]] if labels[i] >= 0 else np.zeros_like(f)
        conf.append(np.exp(-np.linalg.norm(f - c)))
    return np.array(conf)

def update_memory_bank(memory_bank, feats, labels, conf_scores, conf_thresh=0.7):
    for i, score in enumerate(conf_scores):
        if score > conf_thresh:
            memory_bank.update(feats[i], labels[i], conf=score)