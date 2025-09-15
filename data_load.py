import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class ReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_source=True, attr_extractor=None, device='cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.is_source = is_source
        self.image_paths = []
        self.labels = []
        self.cam_ids = []
        self.frame_ids = []
        self.attr_extractor = attr_extractor
        self.device = device

        if not os.path.isdir(root_dir):
            raise ValueError(f"Dataset directory {root_dir} not found!")

        valid_images = [img for img in os.listdir(root_dir) if img.endswith('.jpg')]
        print(f"\nInitializing {'source' if is_source else 'target'} dataset:")
        print(f"Found {len(valid_images)} images in {root_dir}")

        for img_name in valid_images:
            try:
                parts = img_name.split('_')
                if self.is_source:
                    if len(parts) == 3 and parts[2].startswith('f'):
                        # DukeMTMC-reID as source
                        label = int(parts[0])
                        cam_id = int(parts[1][1:])
                        frame_id = int(parts[2][1:].split('.')[0])
                    elif len(parts) >= 3:
                        # Market-1501 as source
                        label = int(parts[0])
                        if label == -1:
                            continue
                        cam_str = parts[1]
                        if 's' in cam_str:
                            cam_id = int(cam_str[1])
                        else:
                            cam_id = int(cam_str[1:]) if cam_str[1:].isdigit() else 0
                        frame_id = int(parts[2]) if parts[2].isdigit() else 0
                    else:
                        raise ValueError("Unknown filename format")
                else:
                    label = -1
                    if len(parts) == 3 and parts[2].startswith('f'):
                        cam_id = int(parts[1][1:]) if parts[1][1:].isdigit() else 0
                        frame_id = int(parts[2][1:].split('.')[0]) if parts[2][1:].split('.')[0].isdigit() else 0
                    elif len(parts) >= 3:
                        cam_str = parts[1]
                        if 's' in cam_str:
                            cam_id = int(cam_str[1])
                        else:
                            cam_id = int(cam_str[1:]) if cam_str[1:].isdigit() else 0
                        frame_id = int(parts[2]) if parts[2].isdigit() else 0
                    else:
                        cam_id = 0
                        frame_id = 0

                self.image_paths.append(os.path.join(root_dir, img_name))
                self.labels.append(label)
                self.cam_ids.append(cam_id)
                self.frame_ids.append(frame_id)
            except (IndexError, ValueError) as e:
                print(f"Skipping invalid file {img_name}: {str(e)}")
                continue

        self.labels = np.array(self.labels)
        if self.is_source:
            unique_labels = np.unique(self.labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            self.labels = np.array([label_map[l] for l in self.labels])
        self.cam_ids = np.array(self.cam_ids)
        self.frame_ids = np.array(self.frame_ids)

        if not self.is_source:
            print(f"Loaded {len(self)} unlabeled target samples")
            if len(self) == 0:
                raise RuntimeError("No valid target samples found!")
        else:
            unique_labels = np.unique(self.labels)
            print(f"Source contains {len(unique_labels)} identities")
        print(f"Sample paths: {self.image_paths[:3]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Extract attribute embedding if extractor is provided
        attr_emb = None
        if self.attr_extractor is not None:
            attributes = self.attr_extractor.extract_attributes(img_path)
            attr_emb = torch.tensor(self.attr_extractor.encode_attributes(attributes), dtype=torch.float32)
        sample = {
            'image': image,
            'label': self.labels[idx],
            'cam_id': self.cam_ids[idx],
            'frame_id': self.frame_ids[idx],
            'path': img_path,
        }
        if attr_emb is not None:
            sample['attr_emb'] = attr_emb
        return sample

def build_transforms(img_size=(224, 224), is_train=True):
    if is_train:
        transforms = [
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transforms = [
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    return T.Compose(transforms)

def build_dataloaders(source_dir, target_dir, batch_size=16, attr_extractor=None, device='cpu'):
    source_train = ReIDDataset(
        root_dir=os.path.join(source_dir, 'bounding_box_train'),
        transform=build_transforms(is_train=True),
        is_source=True,
        attr_extractor=attr_extractor,
        device=device
    )

    target_train = ReIDDataset(
        root_dir=os.path.join(target_dir, 'bounding_box_train'),
        transform=build_transforms(is_train=True),
        is_source=False,
        attr_extractor=attr_extractor,
        device=device
    )

    source_val = ReIDDataset(
        root_dir=os.path.join(source_dir, 'bounding_box_test'),
        transform=build_transforms(is_train=False),
        is_source=True,
        attr_extractor=attr_extractor,
        device=device
    )

    return (
        DataLoader(source_train, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(target_train, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(source_val, batch_size, shuffle=False, num_workers=4)
    )