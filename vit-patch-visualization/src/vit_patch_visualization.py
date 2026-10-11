"""
Vision Transformer Patch Visualization
--------------------------------------
Analyses how the patch size of a Vision Transformer (ViT) changes
  1. the number of tokens,
  2. the spatial information that is kept, and
  3. the feature representation of each patch.

Only NumPy, Pillow and Matplotlib are used, so it runs on any laptop (no GPU,
no internet, no pre-trained weights needed).

Run from the project root:
    python src/vit_patch_visualization.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
IMAGE_PATH = "dataset/sample_image.png"
RESULTS_DIR = "results"
IMG_SIZE = 224                      # standard ViT input size
PATCH_SIZES = [8, 16, 32, 56]       # all divide 224 exactly
EMBED_DIM = 64                      # size of the patch embedding vector
SEED = 42


# ----------------------------------------------------------------------------
# Core ViT steps
# ----------------------------------------------------------------------------
def load_image(path, size=IMG_SIZE):
    """Load an image and resize it to size x size (RGB, values 0-1)."""
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(img, dtype=np.float32) / 255.0


def split_into_patches(image, patch_size):
    """
    Cut the image into non-overlapping square patches and flatten each one.
    Returns an array of shape (num_tokens, patch_size * patch_size * 3).
    """
    h, w, c = image.shape
    gh, gw = h // patch_size, w // patch_size          # patch grid
    patches = image.reshape(gh, patch_size, gw, patch_size, c)
    patches = patches.transpose(0, 2, 1, 3, 4)          # (gh, gw, p, p, c)
    return patches.reshape(gh * gw, patch_size * patch_size * c), (gh, gw)


def patch_embedding(patches, embed_dim=EMBED_DIM, seed=SEED):
    """
    Linear projection of every flattened patch to an embed_dim vector.
    This is exactly what the ViT 'patch embedding' layer does. The weights are
    random here because we only study the effect of patch size (no training).
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1 / np.sqrt(patches.shape[1]), (patches.shape[1], embed_dim))
    return patches @ w


def add_class_token_and_positions(tokens, seed=SEED):
    """Prepend a [CLS] token and add (random) positional embeddings."""
    rng = np.random.default_rng(seed)
    cls = np.zeros((1, tokens.shape[1]))
    seq = np.vstack([cls, tokens])
    pos = rng.normal(0, 0.02, seq.shape)
    return seq + pos


def pca_to_rgb(features):
    """Reduce features to 3 principal components and scale to 0-1 for display."""
    x = features - features.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comp = x @ vt[:3].T
    comp = (comp - comp.min(axis=0)) / (comp.max(axis=0) - comp.min(axis=0) + 1e-8)
    return comp


def cosine_similarity_map(features, query_idx):
    """Cosine similarity between one query patch and all patches."""
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    return f @ f[query_idx]


def mean_color_reconstruction(image, patch_size):
    """
    Replace every patch by its average colour. This shows how much
    spatial detail is lost when the patch is large.
    """
    h, w, c = image.shape
    gh, gw = h // patch_size, w // patch_size
    blocks = image.reshape(gh, patch_size, gw, patch_size, c)
    means = blocks.mean(axis=(1, 3))                     # (gh, gw, c)
    return np.kron(means, np.ones((patch_size, patch_size, 1)))


def mse_psnr(original, recon):
    mse = float(np.mean((original - recon) ** 2))
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    return mse, psnr


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def draw_grid(ax, image, patch_size, title):
    ax.imshow(image)
    for t in range(0, image.shape[0] + 1, patch_size):
        ax.axhline(t - 0.5, color="white", lw=0.6)
        ax.axvline(t - 0.5, color="white", lw=0.6)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    image = load_image(IMAGE_PATH)

    rows = []         # table of numbers for the report
    store = {}        # per-patch-size results used by the plots

    for p in PATCH_SIZES:
        patches, grid = split_into_patches(image, p)
        emb = patch_embedding(patches)
        seq = add_class_token_and_positions(emb)
        recon = mean_color_reconstruction(image, p)
        mse, psnr = mse_psnr(image, recon)
        n = patches.shape[0]
        rows.append({
            "patch": p,
            "grid": f"{grid[0]}x{grid[1]}",
            "tokens": n,
            "seq_len": seq.shape[0],
            "patch_dim": patches.shape[1],
            "attn_entries": seq.shape[0] ** 2,
            "mse": mse,
            "psnr": psnr,
        })
        store[p] = dict(patches=patches, emb=emb, grid=grid, recon=recon)

    # ---- Figure 1: patch grids ------------------------------------------------
    fig, axes = plt.subplots(1, len(PATCH_SIZES) + 1, figsize=(4 * (len(PATCH_SIZES) + 1), 4.4))
    axes[0].imshow(image); axes[0].set_title("Original 224x224", fontsize=10); axes[0].axis("off")
    for ax, r in zip(axes[1:], rows):
        draw_grid(ax, image, r["patch"],
                  f"Patch {r['patch']}x{r['patch']}\n{r['grid']} grid = {r['tokens']} tokens")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/01_patch_grids.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    # ---- Figure 2: token count and attention cost -------------------------------
    labels = [f"{r['patch']}x{r['patch']}" for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bars = ax[0].bar(labels, [r["tokens"] for r in rows], color="#3b6fb6")
    ax[0].set_title("Number of tokens vs patch size")
    ax[0].set_xlabel("Patch size"); ax[0].set_ylabel("Tokens"); ax[0].set_yscale("log")
    for b, r in zip(bars, rows):
        ax[0].text(b.get_x() + b.get_width() / 2, r["tokens"], str(r["tokens"]),
                   ha="center", va="bottom")
    ax[1].plot(labels, [r["attn_entries"] for r in rows], "o-", color="#c0504d")
    ax[1].set_yscale("log")
    ax[1].set_title("Self-attention matrix size (tokens+CLS)^2")
    ax[1].set_xlabel("Patch size"); ax[1].set_ylabel("Entries per head")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/02_tokens_and_attention_cost.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    # ---- Figure 3: spatial information (mean-colour reconstruction) -------------
    fig, axes = plt.subplots(1, len(PATCH_SIZES) + 1, figsize=(4 * (len(PATCH_SIZES) + 1), 4.4))
    axes[0].imshow(image); axes[0].set_title("Original", fontsize=10); axes[0].axis("off")
    for ax, r in zip(axes[1:], rows):
        ax.imshow(store[r["patch"]]["recon"])
        ax.set_title(f"Patch {r['patch']}  PSNR={r['psnr']:.1f} dB", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/03_spatial_information.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    # ---- Figure 4: feature representation (PCA of patch embeddings -> RGB) ------
    fig, axes = plt.subplots(1, len(PATCH_SIZES), figsize=(4 * len(PATCH_SIZES), 4.2))
    for ax, p in zip(np.atleast_1d(axes), PATCH_SIZES):
        gh, gw = store[p]["grid"]
        rgb = pca_to_rgb(store[p]["emb"]).reshape(gh, gw, 3)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"Patch {p}: PCA of embeddings\n({gh}x{gw} tokens)", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/04_feature_representation_pca.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    # ---- Figure 5: patch similarity map for a query patch -----------------------
    # Query = the patch containing the image point (row 70, col 110) (around the face).
    qy, qx = 70, 110
    fig, axes = plt.subplots(1, len(PATCH_SIZES), figsize=(4 * len(PATCH_SIZES), 4.2))
    for ax, p in zip(np.atleast_1d(axes), PATCH_SIZES):
        gh, gw = store[p]["grid"]
        q = (qy // p) * gw + (qx // p)
        sim = cosine_similarity_map(store[p]["emb"], q).reshape(gh, gw)
        ax.imshow(image)
        ax.imshow(np.kron(sim, np.ones((p, p))), cmap="jet", alpha=0.55, vmin=-1, vmax=1)
        ax.add_patch(plt.Rectangle(((qx // p) * p - 0.5, (qy // p) * p - 0.5), p, p,
                                   fill=False, ec="white", lw=2))
        ax.set_title(f"Patch {p}: similarity to query patch", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/05_patch_similarity_map.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    # ---- Figure 6: a few individual patches (what one token actually sees) ------
    fig, axes = plt.subplots(len(PATCH_SIZES), 6, figsize=(9, 1.6 * len(PATCH_SIZES)))
    rng = np.random.default_rng(0)
    for i, p in enumerate(PATCH_SIZES):
        patches = store[p]["patches"]
        for j, idx in enumerate(rng.choice(len(patches), 6, replace=False)):
            axes[i, j].imshow(patches[idx].reshape(p, p, 3))
            axes[i, j].axis("off")
        axes[i, 0].set_title(f"{p}x{p}", fontsize=9, loc="left")
    plt.suptitle("Sample individual patches (one patch = one token)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/06_sample_patches.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    # ---- Results table -----------------------------------------------------------
    header = ("Patch | Grid  | Tokens | Seq len (+CLS) | Values per patch | "
              "Attention entries | MSE      | PSNR (dB)")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['patch']:>5} | {r['grid']:<5} | {r['tokens']:>6} | {r['seq_len']:>14} | "
                     f"{r['patch_dim']:>16} | {r['attn_entries']:>17} | "
                     f"{r['mse']:.5f} | {r['psnr']:.2f}")
    table = "\n".join(lines)
    print(table)
    with open(f"{RESULTS_DIR}/results_table.txt", "w") as f:
        f.write(table + "\n")

    print(f"\nFigures and table saved in '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
