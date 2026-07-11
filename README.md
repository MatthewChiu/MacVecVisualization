# Composite YCAC — Macroharmony Embedding Explorer

A self-contained, static web app for exploring the `compositeYCACModel_TonicRanked` word2vec
embeddings: 32,916 macroharmony vectors (32-dim), each keyed by a 12-slot pitch-class vector
(count of each scale degree, 0 = tonic, ranked relative to the local key) mined from the
Yale–Classical Archives Corpus.

It works like the TensorFlow Embedding Projector: a rotatable/pannable point cloud you can
search, filter, and click into for nearest neighbors — but everything runs client-side, with
no server or build step.

## What's inside

```
index.html      the page
style.css       styling
app.js          three.js scene, filters, search, nearest-neighbor logic
assets/         favicon
data/
  data.json     per-macroharmony metadata (pitch-class vector, size, frequency, tonic weight)
  vectors.bin   raw 32-dim embeddings, Float32Array, row-major (N × 32)
  pca3.bin      3D PCA projection, Float32Array (N × 3), pre-normalized to [-1, 1]
  tsne2.bin     2D t-SNE projection, Float32Array (N × 2), pre-normalized to [-1, 1]
```

The projections and metadata were precomputed offline from your `.model` file (a pickled
gensim `Word2Vec`) with scikit-learn's PCA and t-SNE — there's no gensim dependency at
runtime, only static binary/JSON files.

## Running locally

Because the app `fetch()`es the `data/` files, you can't just double-click `index.html`
(browsers block `fetch` on `file://`). Serve the folder instead:

```bash
cd path/to/this/folder
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploying to GitHub Pages, then mattchiu.com

1. Push this folder to a GitHub repo (e.g. as a subfolder like `macroharmony-embeddings/` in your
   site repo, or its own repo).
2. In the repo's Settings → Pages, set the source to the branch/folder containing these files.
3. Once Pages is live, you can either:
   - link out to the Pages URL from mattchiu.com, or
   - point a subdomain/path at it if your host supports proxying, or
   - copy this folder directly into wherever mattchiu.com's static files are served from —
     it has no server-side dependencies, so it will work anywhere static files are served.

No API keys, no build tooling, no CORS concerns beyond the two CDN font/three.js requests
(Google Fonts + jsdelivr), which work from any domain.

## How the visuals map to the model

- **PCA · 3D** — the top 3 principal components of the raw 32-dim embedding. Rotate freely.
- **t-SNE · 2D** — a 2D t-SNE layout (perplexity 30), generally shows tighter local clusters
  at the expense of global distances. Rotation is disabled in this mode; pan/zoom only.
- **Color** — fixed to macroharmony size (how many distinct scale degrees sound), teal (small)
  to violet (large); see the legend under the size filter.
- **Filters** — chord size, minimum frequency, free-text search, and a **Collections** row of
  presets (all macroharmonies, most common, diatonic scales). Points that don't pass the
  current filter are dimmed and inert — hovering or clicking them does nothing until you widen
  the filter again.
- **Pitch-class wheel** (detail panel) — a 12-slot radial chart of the selected macroharmony's
  vector, tonic marked in brass at 12 o'clock.
- **Nearest neighbors** — cosine similarity computed live in-browser over the full 32-dim
  vectors (not the 2D/3D projection), so it reflects the model's actual geometry.

## Regenerating the data files

If you retrain the model or want to adjust the projections, the pipeline that produced
`data/` was:

1. Unpickle the gensim `Word2Vec`, pull out `.wv.index_to_key`, `.wv.vectors`, and
   `.wv.expandos['count']`.
2. Parse each key (a string like `"[0, 1, 2, ...]"`) back into its 12-int list.
3. Fit `sklearn.decomposition.PCA(n_components=3)` and `sklearn.manifold.TSNE(n_components=2,
   perplexity=30, init="pca")` on the 32-dim vectors.
4. Normalize each projection to `[-1, 1]` per axis and write everything out as the binary/JSON
   files above.

Happy to hand over that script if you want to swap in a different model or a UMAP layout.
