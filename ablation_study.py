import torch
import random
import numpy as np
import logging
from mutual_models import build_models as build_maformer
from trainer import UDATrainer
from data_load import build_dataloaders

# ==== ViT Implementation for Baselines ====
from transformers import ViTModel
import torch.nn as nn

class SimpleViTClassifier(nn.Module):
    def __init__(self, num_classes, checkpoint_path='/home/tq_naeem/Project/.venv/main/'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(checkpoint_path)
        self.head = nn.Linear(768, num_classes)
    def forward(self, img, attr_emb=None, return_features=False):
        vis_out = self.vit(img).last_hidden_state
        pooled = vis_out[:, 0]
        if return_features:
            return pooled
        return self.head(pooled)

# ==== Pseudo-labeling hooks ====
from pseudo_labeling import (
    extract_features, run_dbscan, compute_confidence, update_memory_bank
)

# ==== Runtime Config ====
SOURCE_DIR = "/home/tq_naeem/Project/.venv/main/DukeMTMC-reID"
TARGET_DIR = "/home/tq_naeem/Project/.venv/main/datasets/Market-1501-v15.09.15"
NUM_CLASSES = 702
BATCH_SIZE = 16
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluate_placeholder(trainer):
    
   def evaluate(self):
        """Evaluate model(s) on the target validation set and print results."""
        import torch
        import numpy as np
        self.Net_S.eval()
        val_loader = self.val_loader_target
        device = self.device
        total = 0
        correct = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                attr = batch.get('attr_emb', None)
                if attr is not None:
                    attr = attr.to(device)
                label = batch['label'].to(device)
                if attr is not None:
                    outputs = self.Net_S(img, attr)
                else:
                    outputs = self.Net_S(img, None)
                loss = criterion(outputs, label)
                total_loss += loss.item() * img.size(0)
                _, preds = torch.max(outputs, 1)
                total += label.size(0)
                correct += (preds == label).sum().item()
                all_preds.append(preds.cpu())
                all_labels.append(label.cpu())
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        print(f"[EVAL] Rank-1 Accuracy: {accuracy*100:.2f}% | Avg. loss: {avg_loss:.4f}")
        return accuracy, avg_loss

# ==== Experiment Routines ====

def run_baseline_vit(device):
    logging.info("Running Baseline (Only ViT)...")
    set_seed(SEED)
    vit = SimpleViTClassifier(NUM_CLASSES).to(device)
    attr_extractor = None
    train_loader_source, _, val_loader_target = build_dataloaders(
        source_dir=SOURCE_DIR, target_dir=TARGET_DIR, batch_size=BATCH_SIZE,
        attr_extractor=attr_extractor, device=device
    )
    trainer = UDATrainer(vit, None, None, (train_loader_source, None, val_loader_target), device)
    trainer.pretrain_source(epochs=10)
    if hasattr(trainer, "evaluate"):
        trainer.evaluate()
    else:
        evaluate_placeholder(trainer)

def run_vit_pseudo_labels(device):
    logging.info("Running ViT + Pseudo Labels...")
    set_seed(SEED)
    vit = SimpleViTClassifier(NUM_CLASSES).to(device)
    attr_extractor = None
    train_loader_source, train_loader_target, val_loader_target = build_dataloaders(
        source_dir=SOURCE_DIR, target_dir=TARGET_DIR, batch_size=BATCH_SIZE,
        attr_extractor=attr_extractor, device=device
    )
    trainer = UDATrainer(vit, None, None, (train_loader_source, train_loader_target, val_loader_target), device)
    trainer.pretrain_source(epochs=10)

    # Feature extraction for pseudo-labeling
    feats, _ = extract_features(vit, train_loader_target, device)
    pseudo_labels = run_dbscan(feats)
    valid = pseudo_labels != -1
    centroids = []
    for i in set(pseudo_labels):
        if i == -1: continue
        centroids.append(feats[pseudo_labels == i].mean(0))
    centroids = np.array(centroids)
    conf_scores = compute_confidence(feats, centroids, pseudo_labels)
    if hasattr(trainer, "memory_bank"):
        update_memory_bank(trainer.memory_bank, feats, pseudo_labels, conf_scores)

    # Adaptation loop
    for epoch in range(40):
        vit.train()
        for batch in train_loader_target:
            img = batch['image'].to(device)
            idxs = batch.get('idx', list(range(len(img))))
            labels = torch.tensor([pseudo_labels[i] if conf_scores[i] > 0.7 else -1 for i in idxs]).to(device)
            if (labels == -1).all(): continue
            out = vit(img)
            mask = labels != -1
            if mask.sum() == 0: continue
            loss = trainer.sce(out[mask], labels[mask]) if hasattr(trainer, "sce") else torch.nn.CrossEntropyLoss()(out[mask], labels[mask])
            trainer.optim_S.zero_grad()
            loss.backward()
            trainer.optim_S.step()
    if hasattr(trainer, "evaluate"):
        trainer.evaluate()
    else:
        evaluate_placeholder(trainer)

def run_maformer_only(device):
    logging.info("Running MAFormer only...")
    set_seed(SEED)
    Net_S, _, _ = build_maformer(num_classes=NUM_CLASSES)
    Net_S = Net_S.to(device)
    attr_extractor = None
    train_loader_source, train_loader_target, val_loader_target = build_dataloaders(
        source_dir=SOURCE_DIR, target_dir=TARGET_DIR, batch_size=BATCH_SIZE,
        attr_extractor=attr_extractor, device=device
    )
    trainer = UDATrainer(Net_S, None, None, (train_loader_source, train_loader_target, val_loader_target), device)
    trainer.pretrain_source(epochs=10)

    feats, _ = extract_features(Net_S, train_loader_target, device)
    pseudo_labels = run_dbscan(feats)
    centroids = []
    for i in set(pseudo_labels):
        if i == -1: continue
        centroids.append(feats[pseudo_labels == i].mean(0))
    centroids = np.array(centroids)
    conf_scores = compute_confidence(feats, centroids, pseudo_labels)
    if hasattr(trainer, "memory_bank"):
        update_memory_bank(trainer.memory_bank, feats, pseudo_labels, conf_scores)

    for epoch in range(40):
        Net_S.train()
        for batch in train_loader_target:
            img = batch['image'].to(device)
            attr = batch['attr_emb'].to(device) if 'attr_emb' in batch else None
            idxs = batch.get('idx', list(range(len(img))))
            labels = torch.tensor([pseudo_labels[i] if conf_scores[i] > 0.7 else -1 for i in idxs]).to(device)
            if (labels == -1).all(): continue
            out = Net_S(img, attr)
            mask = labels != -1
            if mask.sum() == 0: continue
            loss = trainer.sce(out[mask], labels[mask]) if hasattr(trainer, "sce") else torch.nn.CrossEntropyLoss()(out[mask], labels[mask])
            trainer.optim_S.zero_grad()
            loss.backward()
            trainer.optim_S.step()
    if hasattr(trainer, "evaluate"):
        trainer.evaluate()
    else:
        evaluate_placeholder(trainer)

def run_maformer_mupl(device):
    logging.info("Running MAFormer + MUPL (mutual learning)...")
    set_seed(SEED)
    Net_S, Net_T, Net_P = build_maformer(num_classes=NUM_CLASSES)
    Net_S, Net_T, Net_P = Net_S.to(device), Net_T.to(device), Net_P.to(device)
    attr_extractor = None
    train_loader_source, train_loader_target, val_loader_target = build_dataloaders(
        source_dir=SOURCE_DIR, target_dir=TARGET_DIR, batch_size=BATCH_SIZE,
        attr_extractor=attr_extractor, device=device
    )
    trainer = UDATrainer(Net_S, Net_T, Net_P, (train_loader_source, train_loader_target, val_loader_target), device)
    trainer.pretrain_source(epochs=10)
    trainer.unsupervised_adaptation(epochs=40)
    if hasattr(trainer, "evaluate"):
        trainer.evaluate()
    else:
        evaluate_placeholder(trainer)

def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*20, "Baseline (Only ViT)", "="*20)
    run_baseline_vit(device)
    print("="*54)
    print("="*20, "ViT + Pseudo Labels", "="*20)
    run_vit_pseudo_labels(device)
    print("="*54)
    print("="*20, "MAFormer only", "="*20)
    run_maformer_only(device)
    print("="*54)
    print("="*20, "MAFormer + MUPL", "="*20)
    run_maformer_mupl(device)
    print("="*54)

if __name__ == "__main__":
    main()