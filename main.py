import torch
from mutual_models import build_models
from data_load import build_dataloaders
from trainer import UDATrainer
from clip_attribute_extractor import CLIPAttributeExtractor
import torch.multiprocessing as mp
from logging_setup import setup_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_dir = "/home/tq_naeem/Project/.venv/main/DukeMTMC-reID"
    target_dir = "/home/tq_naeem/Project/.venv/main/datasets/Market-1501-v15.09.15"
    attr_extractor = CLIPAttributeExtractor(device=device)
    train_loader_source, train_loader_target, val_loader_target = build_dataloaders(
        source_dir=source_dir, target_dir=target_dir, batch_size=16,
        attr_extractor=attr_extractor, device=device
    )
    Net_S, Net_T, Net_P = build_models(num_classes=702)
    trainer = UDATrainer(Net_S, Net_T, Net_P, (train_loader_source, train_loader_target, val_loader_target), device)
    trainer.pretrain_source(epochs=10)
    trainer.unsupervised_adaptation(epochs=40)
    setup_logger('train_full.log')
   #trainer.evaluate()