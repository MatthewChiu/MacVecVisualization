# Composite YCAC — Chord Embedding Explorer

A companion website for the article, "Studying Scales and Macroharmonies in the Yale Classical Archives Corpus"

A self-contained, static web app for exploring the `compositeYCACModel_TonicRanked` word2vec
embeddings: 32,916 chord vectors (32-dim), each keyed by a 12-dimensional tonic-ranked pitch-class vector tonic-ranked. Vectors represent absent PCs as 0, present PCs are 1, and the tonic (determined by the IQ algorithm), e.g. C major is [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]. Data is derived from 14 composers from the Yale–Classical Archives Corpus.

It works like the TensorFlow Embedding Projector: a rotatable/pannable point cloud you can
search, filter, and click into for nearest neighbors — but everything runs client-side, with
no server or build step.

## Running locally

Because the app `fetch()`es the `data/` files, you can't just double-click `index.html`
(browsers block `fetch` on `file://`). Serve the folder instead:

```bash
cd path/to/this/folder
python3 -m http.server 8000
# open http://localhost:8000
```

## How the visuals map to the model

- **PCA · 3D** — the top 3 principal components of the raw 32-dim embedding. Rotate freely.
- **t-SNE · 2D** — a 2D t-SNE layout (perplexity 30), generally shows tighter local clusters
  at the expense of global distances. Rotation is disabled in this mode; pan/zoom only.
- **Color by** — chord size (how many distinct scale degrees sound), frequency in the corpus,
  or tonic weight (how many times scale-degree 0 is doubled).
- **Pitch-class wheel** (detail panel) — a 12-slot radial chart of the selected chord's vector,
  tonic marked in brass at 12 o'clock.
- **Nearest neighbors** — cosine similarity computed live in-browser over the full 32-dim
  vectors (not the 2D/3D projection), so it reflects the model's actual geometry.
  
