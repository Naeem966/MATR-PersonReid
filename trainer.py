import torch
from losses import SymmetricCrossEntropy, mutual_learning_loss, PartAwareTripletLoss
from pseudo_labeling import *
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
class MemoryBank:
    def __init__(self, size, dim, device):
        self.size = size
        self.device = device
        self.bank = torch.zeros(size, dim).to(device)
        self.labels = -torch.ones(size).long().to(device)
        self.conf = torch.zeros(size).to(device)
        self.ptr = 0

    def update(self, feat, label, conf):
        self.bank[self.ptr] = torch.tensor(feat).to(self.device)
        self.labels[self.ptr] = label
       #self.conf[self.ptr] = conf
        self.conf[self.ptr] = float(conf)
        self.ptr = (self.ptr + 1) % self.size

class UDATrainer:
    def __init__(self, Net_S, Net_T, Net_P, loaders, device):
        self.Net_S, self.Net_T, self.Net_P = Net_S.to(device), Net_T.to(device), Net_P.to(device)
        self.optim_S = torch.optim.AdamW(Net_S.parameters(), lr=3e-4)
        self.optim_T = torch.optim.AdamW(Net_T.parameters(), lr=3e-4)
        self.optim_P = torch.optim.AdamW(Net_P.parameters(), lr=3e-4)
        self.train_loader_source, self.train_loader_target, self.val_loader_target = loaders
        self.device = device
        self.memory_bank = MemoryBank(size=10000, dim=768, device=device)
        self.sce = SymmetricCrossEntropy()
        self.part_triplet = PartAwareTripletLoss()

    def pretrain_source(self, epochs=10):
        self.Net_S.train()
        print("Starting training...")
        logging.info("Starting training...")
        for epoch in range(epochs):
           #print(f"Epoch {epoch+1} started")
            logging.info(f"Epoch {epoch+1} started")
            logging.info("🚦 Starting training epoch...")
            batch_iter = tqdm(enumerate(self.train_loader_source), total=len(self.train_loader_source))
            for batch_idx,batch in enumerate(self.train_loader_source):
               #print(f"Processing batch {batch_idx+1}/{len(self.train_loader_source)}")
                img, attr, label = batch['image'].to(self.device), batch['attr_emb'].to(self.device), batch['label'].to(self.device)
                out = self.Net_S(img, attr)
                loss = self.sce(out, label)
                self.optim_S.zero_grad()
                loss.backward()
                self.optim_S.step()
                batch_iter.set_description(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader_source)} | Loss: {loss.item():.4f}")
                logging.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader_source)} | Loss: {loss.item():.4f}")
                logging.info("Forward pass...")
                logging.info("Contrastive loss calculation...")
                logging.info(f"Batch {batch_idx+1} completed | Loss: {loss.item():.4f}")
    def log_test_results(self, epoch, loss, rank1, rank5, mAP):
        logging.info(f"Epoch {epoch} Test Results | Loss: {loss:.4f} | Rank-1: {rank1:.4f} | Rank-5: {rank5:.4f} | mAP: {mAP:.4f}")
    def unsupervised_adaptation(self, epochs=40):
        for epoch in range(epochs):
            feats_S, _ = extract_features(self.Net_S, self.train_loader_target, self.device)
            feats_T, _ = extract_features(self.Net_T, self.train_loader_target, self.device)
            feats_P, _ = extract_features(self.Net_P, self.train_loader_target, self.device)
            pseudo_labels = run_dbscan(feats_S)
            consensus = consensus_topk_neighbors([feats_S, feats_T, feats_P], k=5)
            centroids = np.array([feats_S[pseudo_labels == i].mean(0) for i in set(pseudo_labels) if i != -1])
            conf_scores = compute_confidence(feats_S, centroids, pseudo_labels)
            update_memory_bank(self.memory_bank, feats_S, pseudo_labels, conf_scores)
            self.Net_S.train()
            self.Net_T.train()
            self.Net_P.train()
            for batch in self.train_loader_target:
                img, attr = batch['image'].to(self.device), batch['attr_emb'].to(self.device)
                labels = torch.tensor([pseudo_labels[i] if conf_scores[i] > 0.7 else -1 for i in range(len(pseudo_labels))]).to(self.device)
                if (labels == -1).all(): continue
                out_S = self.Net_S(img, attr)
                out_T = self.Net_T(img, attr)
                out_P = self.Net_P(img, attr)
                mask = labels != -1
                if mask.sum() == 0: continue
                loss_sce = self.sce(out_S[mask], labels[mask]) + self.sce(out_T[mask], labels[mask]) + self.sce(out_P[mask], labels[mask])
                loss_ml = mutual_learning_loss([out_S, out_T, out_P], [labels, labels, labels])
                loss = loss_sce + 0.5 * loss_ml
                self.optim_S.zero_grad()
                self.optim_T.zero_grad()
                self.optim_P.zero_grad()
                loss.backward()
                self.optim_S.step()
                self.optim_T.step()
                self.optim_P.step()