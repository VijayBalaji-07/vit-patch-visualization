# Vision Transformer Patch Visualization

**Course:** Deep Learning Essentials (21ML53T-26) - Innovative Assignment II
**Topic:** Analyze how different patch sizes affect the number of tokens, spatial information, and feature representation, and interpret the resulting visualizations. (CO4, K4)

## What this project does
A Vision Transformer (ViT) cuts an image into square patches and treats every patch as one token. This project takes one 224x224 image, splits it with four patch sizes (8, 16, 32, 56) and compares:

1. **Number of tokens** and the size of the self-attention matrix.
2. **Spatial information** kept, measured by rebuilding the image from the mean colour of every patch (MSE / PSNR).
3. **Feature representation**, by projecting every patch to a 64-d embedding (the ViT patch-embedding step), then showing the embeddings with PCA colours and a cosine-similarity map.

> The embedding weights are random (no training, no pre-trained model). The goal is to study the effect of patch size, not classification accuracy.

## Folder structure
```
vit-patch-visualization/
|-- README.md
|-- requirements.txt
|-- dataset/            sample_image.png
|-- src/                vit_patch_visualization.py   (main script)
|-- notebooks/          ViT_Patch_Analysis.ipynb     (step-by-step version)
|-- results/            01..06 figures + results_table.txt
|-- screenshots/        output.png
```

## How to run
```bash
git clone https://github.com/<your-username>/vit-patch-visualization.git
cd vit-patch-visualization
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python src/vit_patch_visualization.py
```
Run it from the project root. Figures are written to `results/`. For the notebook: `jupyter notebook notebooks/ViT_Patch_Analysis.ipynb`.

## Results
| Patch | Grid | Tokens | Values per patch | Attention entries (per head) | PSNR (dB) |
|------:|:----:|-------:|-----------------:|-----------------------------:|----------:|
| 8  | 28x28 | 784 | 192  | 616,225 | 17.30 |
| 16 | 14x14 | 196 | 768  | 38,809  | 14.78 |
| 32 | 7x7   | 49  | 3072 | 2,500   | 12.94 |
| 56 | 4x4   | 16  | 9408 | 289     | 12.03 |

Figures in `results/`: patch grids, token count and attention cost, spatial information, PCA of embeddings, patch similarity map, sample patches.

## Analysis (short)
* **Tokens:** tokens = (224 / patch)^2, so halving the patch size gives 4x the tokens and about 16x the attention matrix.
* **Spatial information:** PSNR falls as patches grow, because each patch is squeezed into one token and fine edges disappear.
* **Features:** small patches give smooth, detailed embedding maps where object parts (face, helmet, flag) are separable. Large patches give coarse blocks where one token mixes several objects.
* **Trade-off:** patch 16 (the standard ViT-B/16) balances detail and cost.

## Acknowledgements
* Sample image: `astronaut` image from scikit-image's data module (NASA, public domain).
* Libraries: NumPy, Matplotlib, Pillow, Jupyter.
* Concept: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), 2020.
* The code was written with help from Claude (an AI assistant) and then reviewed and understood by the author.
