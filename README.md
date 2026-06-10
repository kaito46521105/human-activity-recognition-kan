# Few-shot Human Activity Recognition using Smooth Factorized KAN

## 概要
本研究では、スマートフォンの加速度・角速度センサから得られる特徴量を用いて、歩行・階段昇降・着座などの人間行動を分類する Human Activity Recognition（HAR）を対象に、少ない学習データでも高精度に推定できるモデル設計を検討しました。

特に、Kolmogorov-Arnold Networks（KAN）の「各入力特徴に対して1次元の非線形関数を学習し、それらを線形に混合して表現を作る」構造に着目し、特徴ごとの滑らかな関数学習と、特徴間の混合を分離した因子分解構造を導入しました。

## 背景
スマートフォンセンサを用いたHARでは、被験者ごとの動作の違いやセンサの持ち方の違いにより、学習データとテストデータの分布が変化しやすいという課題があります。また、実環境では十分なラベル付きデータを集めることが難しいため、少数データでも汎化しやすいモデルが求められます。

## 提案手法
本研究では、KANを基盤として以下の工夫を取り入れました。

- 特徴ごとに滑らかな1次元非線形関数を学習
- 特徴ごとの変換と特徴間の混合を分離した因子分解構造
- 少数データ下での過学習を抑えるモデル設計

## 実験設定
UCI HARデータセットを用いて評価を行いました。

- データセット：UCI Human Activity Recognition Dataset
- 入力：スマートフォンの加速度・角速度センサに基づく特徴量
- タスク：歩行、階段昇降、着座などの行動分類
- 評価条件：
  - LOSO（Leave-One-Subject-Out）
  - micro-shot / few-shot setting
- 比較対象：
  - MLP（多層パーセプトロン）

## 結果
被験者間の分布差を考慮したLOSO条件、および各クラスごく少数サンプルのみで学習するmicro-shot条件において、一般的なMLPと比較して性能改善の傾向を確認しました。

この結果から、提案したSmooth Factorized KANは、低データかつ分布シフトが存在する条件下で、HARモデルの汎化性能向上に有効である可能性が示されました。

## 使用技術
- Python
- PyTorch
- Machine Learning
- Human Activity Recognition
- Kolmogorov-Arnold Networks
- Few-shot Learning

## ソースコード
# %% [colab] UCI HAR — LOSO + MICRO-SHOT: MLP vs improved KAN
import os, zipfile, urllib.request, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================================================
# Config
# =========================================================
SEED = 1337
TEST_SUBJECT = 1     # 1..30; LOSO held-out subject
K_PER_CLASS = 1      # try 1, 2, 3 for hard few-shot; 5 is less harsh
VAL_FRAC = 0.3       # used only if we can form a val set (K>=2)
BATCH_SIZE = 32
EPOCHS = 400
PATIENCE = 30
LR = 1e-3
WEIGHT_DECAY = 5e-4

# MLP size
H1, H2 = 128, 64
DROP = 0.3

# KAN hyperparams
KAN_WIDTH = 96       # hidden width after first KAN block
KAN_BASIS = 16       # hat basis functions per feature (uniform knots)
KAN_DROP = 0.2
KAN_RANGE = 3.0      # knot range in standardized coords (±KAN_RANGE)
KAN_COMPRESS = 128   # optional linear compression dim before KAN (None to disable)
KAN_SMOOTH_LAMBDA = 1e-3  # smoothness regularization on spline coeffs

# =========================================================
# Reproducibility
# =========================================================
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# =========================================================
# Data: download, LOSO split, micro-shot cap, standardize
# =========================================================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
zip_path = "/content/UCI_HAR.zip"
root_dir = "/content/UCI HAR Dataset"
if not os.path.exists(zip_path):
    print("Downloading UCI HAR zip (~60MB)...")
    with urllib.request.urlopen(URL) as r, open(zip_path, "wb") as f:
        f.write(r.read())
if not os.path.exists(root_dir):
    print("Extracting...")
    zipfile.ZipFile(zip_path).extractall("/content")
else:
    print("Dataset present.")

def load_features_with_subjects(split):
    X = np.loadtxt(os.path.join(root_dir, split, f"X_{split}.txt"))
    y = np.loadtxt(os.path.join(root_dir, split, f"y_{split}.txt")).astype(int) - 1
    subj = np.loadtxt(os.path.join(root_dir, split, f"subject_{split}.txt")).astype(int)
    return X, y, subj

X_tr, y_tr, s_tr = load_features_with_subjects("train")
X_te, y_te, s_te = load_features_with_subjects("test")
X_all = np.concatenate([X_tr, X_te], axis=0)
y_all = np.concatenate([y_tr, y_te], axis=0)
subj_all = np.concatenate([s_tr, s_te], axis=0)
n_classes = len(np.unique(y_all))
d_in = X_all.shape[1]

train_mask = subj_all != TEST_SUBJECT
test_mask  = subj_all == TEST_SUBJECT
X_train0, y_train0 = X_all[train_mask], y_all[train_mask]
X_test0,  y_test0  = X_all[test_mask],  y_all[test_mask]
print(f"LOSO held-out subject {TEST_SUBJECT}: train_pool={X_train0.shape}, test={X_test0.shape}, d={d_in}")

def cap_per_class(X, y, k, seed=SEED):
    rng = np.random.default_rng(seed)
    idxs = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        rng.shuffle(cand)
        idxs.append(cand[:min(k, len(cand))])
    idxs = np.concatenate(idxs)
    rng.shuffle(idxs)
    return X[idxs], y[idxs]

X_cap, y_cap = cap_per_class(X_train0, y_train0, K_PER_CLASS, seed=SEED)
print("Per-class counts in capped train:")
for c in range(n_classes):
    print(f"  class {c}: {(y_cap==c).sum()}")

scaler = StandardScaler().fit(X_cap)
X_tr_fs = scaler.transform(X_cap).astype(np.float32)
X_te_fs = scaler.transform(X_test0).astype(np.float32)

def stratified_split_idx(y, frac, seed=SEED):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        rng.shuffle(cand)
        n_val = 1 if len(cand) >= 2 else 0
        if len(cand) >= 3:
            n_val = max(n_val, int(len(cand)*frac))
        va.append(cand[:n_val]); tr.append(cand[n_val:])
    tr = np.concatenate(tr) if tr else np.array([], dtype=int)
    va = np.concatenate(va) if va else np.array([], dtype=int)
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va

tr_idx, va_idx = stratified_split_idx(y_cap, VAL_FRAC, seed=SEED)
use_validation = len(va_idx) > 0

X_tr_sub, y_tr_sub = X_tr_fs[tr_idx], y_cap[tr_idx]
X_va_sub, y_va_sub = (X_tr_fs[va_idx], y_cap[va_idx]) if use_validation else (np.empty((0, d_in), np.float32), np.array([], int))
print(f"Few-shot sizes → train={X_tr_sub.shape[0]}  val={X_va_sub.shape[0]}  test={X_te_fs.shape[0]}")
if not use_validation:
    print("Note: too tiny for validation; early-stopping on TRAIN loss.")

train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_sub), torch.tensor(y_tr_sub, dtype=torch.long)),
                          batch_size=min(BATCH_SIZE, max(1, len(X_tr_sub))), shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_va_sub), torch.tensor(y_va_sub, dtype=torch.long)),
                          batch_size=min(BATCH_SIZE, max(1, len(X_va_sub))), shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_te_fs), torch.tensor(y_test0, dtype=torch.long)),
                          batch_size=256, shuffle=False)

# =========================================================
# Models: MLP (baseline) and improved KAN
# =========================================================
class MLP(nn.Module):
    def __init__(self, d_in, d_h1=128, d_h2=64, d_out=6, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h1), nn.ReLU(inplace=True), nn.LayerNorm(d_h1), nn.Dropout(p),
            nn.Linear(d_h1, d_h2), nn.ReLU(inplace=True), nn.LayerNorm(d_h2), nn.Dropout(p),
            nn.Linear(d_h2, d_out)
        )
    def forward(self, x):
        return self.net(x)

class HatBasis1D(nn.Module):
    """
    Piecewise-linear 'hat' bases on uniform knots in [-R, R].
    For scalar input x, produces B features with compact support.
    """
    def __init__(self, B=16, R=3.0):
        super().__init__()
        self.B = B
        self.R = R
        ts = torch.linspace(-R, R, B)
        self.register_buffer("knots", ts)

    def forward(self, x):  # x: (...,)
        if self.B == 1:
            return torch.ones(*x.shape, 1, device=x.device, dtype=x.dtype)
        delta = (2 * self.R) / (self.B - 1)
        x = x.unsqueeze(-1)
        dist = torch.abs(x - self.knots)
        phi = torch.clamp(1.0 - dist / delta, min=0.0)
        return phi

class FactorizedKANBlock(nn.Module):
    """
    Factorized KAN block:
      - per-dimension affine normalization of inputs;
      - 1D hat basis on each normalized scalar;
      - per-dimension spline weights A (d x B) to compute univariate functions;
      - linear mixing from these univariate outputs and raw inputs.
    """
    def __init__(self, d_in, H=96, B=16, R=3.0, p=0.2):
        super().__init__()
        self.d_in = d_in
        self.B = B
        self.R = R
        self.basis = HatBasis1D(B=B, R=R)
        # Spline weights per dimension: A ∈ ℝ^{d_in × B}
        self.A = nn.Parameter(torch.zeros(d_in, B))
        # Linear term for each dimension
        self.alpha = nn.Parameter(torch.ones(d_in))
        # Affine transform parameters for basis input
        self.shift = nn.Parameter(torch.zeros(d_in))
        self.log_scale = nn.Parameter(torch.zeros(d_in))
        # Mixing layers
        self.mix_g = nn.Linear(d_in, H)
        self.mix_x = nn.Linear(d_in, H, bias=False)
        self.norm = nn.LayerNorm(H)
        self.drop = nn.Dropout(p)

        nn.init.normal_(self.A, mean=0.0, std=0.1)

    def forward(self, x):
        N, d = x.shape
        assert d == self.d_in
        # Per-dimension affine normalization for the basis
        scale = torch.exp(self.log_scale) + 1e-3
        x_norm = (x - self.shift) / scale    # (N, d)
        # Hat basis on each scalar
        x_flat = x_norm.reshape(-1)          # (N*d,)
        phi_flat = self.basis(x_flat)        # (N*d, B)
        phi = phi_flat.view(N, d, self.B)    # (N, d, B)
        # Univariate spline outputs per dim: u_{n,i} = sum_j A_{i,j} φ_{n,i,j}
        u = torch.einsum("ndb,db->nd", phi, self.A)  # (N, d)
        # Add linear term α_i x_i
        g = u + self.alpha * x               # (N, d)
        # Mix g(x) and raw x
        h = self.mix_g(g) + self.mix_x(x)    # (N, H)
        h = self.norm(h)
        h = torch.relu(h)
        h = self.drop(h)
        return h

class SimpleKANv2(nn.Module):
    """
    Improved KAN model:
      - optional linear compression to d0 dims;
      - two FactorizedKANBlocks;
      - linear head + global linear skip from input.
    """
    def __init__(self, d_in, d_hidden=96, d_out=6, B=16, R=3.0, p=0.2, d_compress=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_compress = d_compress

        if d_compress is not None:
            self.compress = nn.Linear(d_in, d_compress)
            d0 = d_compress
        else:
            self.compress = None
            d0 = d_in

        self.block1 = FactorizedKANBlock(d0, H=d_hidden, B=B, R=R, p=p)
        self.block2 = FactorizedKANBlock(d_hidden, H=d_hidden, B=max(4, B // 2), R=R, p=p)
        self.head = nn.Linear(d_hidden, d_out)
        # Global linear skip from input
        self.head_skip = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        if self.compress is not None:
            z = self.compress(x)
        else:
            z = x
        h = self.block1(z)
        h = self.block2(h)
        logits = self.head(h) + self.head_skip(x)
        return logits

# =========================================================
# Train / eval utilities (shared)
# =========================================================
def run_epoch(model, loader, criterion, optimizer=None, smooth_lambda=0.0):
    train = optimizer is not None
    model.train() if train else model.eval()
    losses, preds, gts = [], [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)

            # Add smoothness regularization for KAN if enabled
            if train and smooth_lambda > 0.0:
                smooth_loss = 0.0
                for m in model.modules():
                    if isinstance(m, FactorizedKANBlock):
                        A = m.A  # (d_in, B)
                        if A.size(1) > 1:
                            diff = A[:, 1:] - A[:, :-1]
                            smooth_loss = smooth_loss + (diff ** 2).mean()
                loss = loss + smooth_lambda * smooth_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            losses.append(loss.item())
            preds.append(logits.argmax(1).detach().cpu().numpy())
            gts.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([], dtype=int)
    gts   = np.concatenate(gts)   if gts   else np.array([], dtype=int)
    acc = accuracy_score(gts, preds) if len(gts) else 0.0
    f1m = f1_score(gts, preds, average="macro", zero_division=0) if len(gts) else 0.0
    return float(np.mean(losses)), acc, f1m

def train_and_eval(model, name, smooth_lambda=0.0):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state, best_score, wait = None, (-np.inf if use_validation else float("inf")), 0
    metric_name = "val_f1" if use_validation else "train_loss"

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer,
                                           smooth_lambda=smooth_lambda)
        if use_validation:
            va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, optimizer=None,
                                               smooth_lambda=0.0)
            score = va_f1
            improved = score > best_score + 1e-4
        else:
            va_loss = va_acc = va_f1 = 0.0
            score = tr_loss
            improved = score < best_score - 1e-4

        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_score, wait = score, 0
        else:
            wait += 1

        print(f"[{name}] Ep {ep:03d} | tr loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
              f"va loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f} "
              f"{'[*]' if improved else ''}")

        if wait >= PATIENCE:
            print(f"[{name}] Early stopping on {metric_name}.")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    te_loss, te_acc, te_f1 = run_epoch(model, test_loader, criterion, optimizer=None,
                                       smooth_lambda=0.0)
    print(f"\n=== {name} — Test (LOSO subj={TEST_SUBJECT}, K/class={K_PER_CLASS}) ===")
    print(f"Loss: {te_loss:.4f} | Acc: {te_acc:.4f} | Macro-F1: {te_f1:.4f}\n")
    return model, (te_loss, te_acc, te_f1)

# =========================================================
# Run both models: MLP baseline, then improved KAN
# =========================================================
mlp = MLP(d_in=d_in, d_h1=H1, d_h2=H2, d_out=n_classes, p=DROP)
mlp, mlp_metrics = train_and_eval(mlp, "MLP", smooth_lambda=0.0)

kan = SimpleKANv2(d_in=d_in, d_hidden=KAN_WIDTH, d_out=n_classes,
                  B=KAN_BASIS, R=KAN_RANGE, p=KAN_DROP,
                  d_compress=KAN_COMPRESS)
kan, kan_metrics = train_and_eval(kan, "KAN", smooth_lambda=KAN_SMOOTH_LAMBDA)

# =========================================================
# Side-by-side summary and confusion matrices
# =========================================================
def eval_and_report(model, name):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            gts.append(yb.numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)

    print(f"Classification report — {name}:")
    print(classification_report(
        gts, preds, digits=4,
        labels=list(range(6)),
        target_names=[
            "WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS",
            "SITTING","STANDING","LAYING"
        ]
    ))
    cm = confusion_matrix(gts, preds, labels=list(range(6)))
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix — {name}")
    plt.xticks(range(6), ["WALK","UP","DOWN","SIT","STAND","LAY"], rotation=45)
    plt.yticks(range(6), ["WALK","UP","DOWN","SIT","STAND","LAY"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); plt.show()

print("Summary (Acc / Macro-F1):")
print(f"  MLP: acc={mlp_metrics[1]:.4f}  f1={mlp_metrics[2]:.4f}")
print(f"  KAN: acc={kan_metrics[1]:.4f}  f1={kan_metrics[2]:.4f}")

eval_and_report(mlp, "MLP")
eval_and_report(kan, "KAN")

