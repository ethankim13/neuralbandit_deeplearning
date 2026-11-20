"""
turns each raw prompt into a fixed-size context vector for the bandit

"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

class TfidfEncoder:
  def __init__(self, max_features = 5000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    self.vec = TfidfVectorizer(lowercase = True, stop_words = "english", ngram_range = (1, 2),
                              min_df = 2, max_features = max_features)

  def fit(self, texts):
    self.vec.fit(list(texts))
    print(f"[tfidf] vocabulary: {len(self.vec.vocabulary_)} terms")
    return self
  
  def transform(self, texts):
    return self.vec.transform(list(texts)).toarray().astype(np.float32)

class PCAReducer:
  """PCA is a linear autoencoder with MSE loss, solved in closed form"""

  def __init__(self, dim = 64, seed = 0):
    self.dim, self.seed = dim, seed
  
  def fit(self, X):
    from sklearn.decomposition import PCA
    if self.dim > min(X.shape):
      raise ValueError(f"dim = {self.dim} is larger than min(n_fit, n_features) = {min(X.shape)}")
    self.pca = PCA(n_components = self.dim, random_state = self.seed).fit(X)
    print(f"[pca] {self.dim} components keep {self.pca.explained_variance_ratio_.sum():.1%} of the variance")
    return self
  
  def transform(self, X):
    return self.pca.transform(X).astype(np.float32)

class DAEReducer:
  """Denoising autoencoder used as a learned, non-linear PCA"""

  def __init__(self, dim = 64, hidden = 256, mask_p = 0.2, lr = 1e-3, weight_decay = 1e-4,
              batch_size = 64, max_epochs = 500, patience = 20, val_frac = 0.15, seed = 0):
    self.dim, self.hidden, self.mask_p = dim, hidden, mask_p
    self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
    self.max_epochs, self.patience, self.val_frac, self.seed = max_epochs, patience, val_frac, seed
  
  def fit(self, X):
    import torch
    from torch import nn
    torch.manual_seed(self.seed)
    self.mu, self.sd = X.mean(0), X.std() + 1e-8
    X = torch.as_tensor((X - self.mu) / self.sd, dtype = torch.float32)
    perm = torch.randperm(len(X))
    n_val = max(1, int(self.val_frac * len(X)))
    X_val, X_tr = X[perm[:n_val]], X[perm[n_val:]]
    X_val_noisy = X_val * (torch.rand_like(X_val) > self.mask_p)  # fixed noisy validation set

    D = X.shape[1]
    self.encoder = nn.Sequential(nn.Linear(D, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.dim))
    decoder = nn.Sequential(nn.Linear(self.dim, self.hidden), nn.ReLU(), nn.Linear(self.hidden, D))
    opt = torch.optim.Adam([*self.encoder.parameters(), *decoder.parameters()],
                          lr = self.lr, weight_decay = self.weight_decay)
    mse = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(1, self.max_epochs + 1):
      for idx in torch.randperm(len(X_tr)).split(self.batch_size):
        x = X_tr[idx]
        x_noisy = x * (torch.rand_like(x) > self.mask_p)
        loss = mse(decoder(self.encoder(x_noisy)), x)
        opt.zero_grad()
        loss.backward()
        opt.step()

      with torch.no_grad():
        val = mse(decoder(self.encoder(X_val_noisy)), X_val).item()

      if val < best:
        best, wait = val, 0
        best_state = {k: v.clone() for k, v in self.encoder.state_dict().items()}
      elif (wait := wait + 1) >= self.patience:
        break
    
    self.encoder.load_state_dict(best_state)  # the decoer is discarded here
    print(f"[dae] stopped after {epoch} epochs, held out noisy MSE {best:.3f} "
          f"(predicting the mean scores = {X_val.pow(2).mean().item():.3f})")
    
    return self

  def transform(self, X):
    import torch
    with torch.no_grad():
      X = torch.as_tensor((X - self.mu) / self.sd, dtype = torch.float32)
      return self.encoder(X).numpy()
  
REDUCERS = {"pca": PCAReducer, "dae": DAEReducer}


# helper methods
def encode_texts(texts, reduce = "pca", dim = 64, fit_idx = None, seed = 0):
  """return a (len(texts), dim) float32 array. TF-IDF and the reducer are fit on texts[fit_idx]"""

  texts = list(texts)
  fit_idx = np.arange(len(texts)) if fit_idx is None else np.asarray(fit_idx)
  enc = TfidfEncoder().fit([texts[i] for i in fit_idx])
  X = enc.transform(texts)
  return REDUCERS[reduce](dim = dim, seed = seed).fit(X[fit_idx]).transform(X).astype(np.float32)
  
def load_embeddings(path = "data/embedddings.npy", ids = None):
  """load cached vectors. pass the prompt ids you're using to catch a stale cache."""

  path = Path(path)
  X = np.load(path)
  meta = json.loads(path.with_suffix(".meta.json").read_text())
  if ids is not None and list(ids) != meta["ids"]:
    raise ValueError(f"{path} was built from a different prompts.parquet. Try re-running sentence_encoder.py")
    
  return X, meta

def main():
  p = argparse.ArgumentParser(description = "Embed data/prompts.parquet with TF-IDF and cache the vectors.")
  p.add_argument("--prompts", default = "data/prompts.parquet")
  p.add_argument("--out", default = "data/embeddings.npy")
  p.add_argument("--reduce", choices = list(REDUCERS), default = "pca")
  p.add_argument("--dim", type = int, default = 64, help = "size of each output vector")
  p.add_argument("--fit-ids", help = "file with one prompt id per line, fit TF-IDF/reducer on these only")
  p.add_argument("--seed", type = int, default = 0)
  a = p.parse_args()

  df = pd.read_parquet(a.prompts)
  fit_idx = None
  if a.fit_ids:
    wanted = set(Path(a.fit_ids).read_text().split())
    fix_idx = np.flatnonzero(df["id"].isin(wanted))
    if len(fit_idx) < len(wanted):
      print(f"warning: {len(wanted) - len(fit_idx)} ids in {a.fit_ids} are not in {a.prompts}")
    
  X = encode_texts(df["prompt"], a.reduce, a.dim, fit_idx, a.seed)

  out = Path(a.out)
  out.parent.mkdir(parents = True, exist_ok = True)
  np.save(out, X)
  meta = {"ids": df["id"].tolist(), "encoder": "tfidf", "reduce": a.reduce, "dim": int(X.shape[1]),
          "n_fit": len(df) if fit_idx is None else int(len(fit_idx)), "seed": a.seed}
  out.with_suffix(".meta.json").write_text(json.dumps(meta, indent = 1))
  print(f"saved {X.shape[0]} x {X.shape[1]} -> {out}")
  
if __name__ == "__main__":
  main()