# MATR-PersonReid

**MATR-PersonReid** implements a state-of-the-art unsupervised domain adaptation (UDA) framework for person re-identification (ReID) across different datasets.
## Overview

Person re-identification aims to match images of the same individual captured by different cameras or under varying conditions. This repository proposes a novel approach for cross-domain person ReID, enabling effective transfer of knowledge from a labeled source domain to an unlabeled target domain.

### Key Features

- **MAFormer (Dynamic Multimodal Transformer):**
  - Splits person images into patches using a Vision Transformer (ViT).
  - Fuses visual features with semantic attribute embeddings derived from natural language descriptions of person attributes (e.g., clothing color, style, accessories).
  - Applies region-specific attention to address occlusions and viewpoint variations.
  - Combines global and local features for robust representation.

- **MUPL (Mutual Update Pseudo-Labeling):**
  - Generates initial pseudo-labels for target domain using DBSCAN clustering.
  - Refines pseudo-labels via multi-model (NetS, NetT, NetP) consensus and k-reciprocal nearest neighbors.
  - Utilizes a memory bank to store high-confidence samples and iteratively relabels hard samples.
  - Employs symmetric cross-entropy and mutual learning losses to improve label robustness.
 **Install dependencies:**
   - Python 3.8+
   - PyTorch
   - torchvision
   - transformers (for ViT and Sentence-BERT)
   - scikit-learn
   - PIL
   - tqdm
   - [CLIP](https://github.com/openai/CLIP)

   Install using pip:
   ```bash
   pip install torch torchvision transformers scikit-learn pillow tqdm
   ```

3. **Download pre-trained models used in the Framework:**
   - Vision Transformer (ViT)
   - Sentence-BERT
   - CLIP
4. Download the model file
https://drive.google.com/file/d/16_BoHBNJ5iBlwBeaVQwPhsahJmxzuoBQ/view?usp=sharing

