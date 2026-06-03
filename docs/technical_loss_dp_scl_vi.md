# Tai Lieu Ky Thuat: Loss Ablation cho DP-SCL

## 1. Muc Dich

Tai lieu nay mo ta thuc nghiem loss ablation vua chay trong
`run_scripts/run_dp_scl_loss_ablation.py`. Thuc nghiem gom ba cau hinh:

| Cau hinh | Duong forward | Loss |
|---|---|---|
| `bce_only` | mot nhanh, khong augmentation, khong projection head | `L_BCE` |
| `supcon_only` | hai nhanh Siamese, co augmentation, co projection head | `L_SupCon` |
| `combined` / DP-SCL | hai nhanh Siamese, co augmentation, co projection head | `L_BCE + lambda * L_SupCon` |

Muc tieu la tach rieng dong gop cua classification loss va supervised
contrastive loss trong khung DP-SCL.

## 2. Tham So Dau Vao

Lenh chay mac dinh cua `run_dp_scl_loss_ablation_fixed.sh`:

```bash
./run_scripts/run_dp_scl_loss_ablation_fixed.sh
```

Sinh ra lenh Python tuong duong:

```bash
python -u run_scripts/run_dp_scl_loss_ablation.py \
  -indir . \
  -outdir . \
  --dataset xuetangx \
  --seeds 1 11 111 1111 11111 \
  --split 0.60 0.10 0.30 \
  --max-epochs 200 \
  --patience 30 \
  --batch-size 64 \
  --lr 1e-4 \
  --hidden-size 128 \
  --lambda-con 0.1 \
  --temperature 0.07 \
  --proposed-mode siamese_lstm_attn \
  --mask-ratio 0.15 \
  --noise-std 0.05 \
  --num-layers 1 \
  --cls-layers 1 \
  --num-workers 0 \
  --run-name dp_scl_loss_ablation_full \
  --force
```

Bang tham so:

| Tham so | Gia tri | Y nghia |
|---|---:|---|
| `dataset` | `xuetangx` | Bo du lieu XuetangX/KDD Cup 2015 |
| `seeds` | `1, 11, 111, 1111, 11111` | 5 lan chia train/val/test doc lap |
| `split` | `0.60/0.10/0.30` | Train/validation/test stratified split |
| `max_epochs` | `200` | So epoch toi da |
| `patience` | `30` | Early stopping theo validation AUC |
| `batch_size` | `64` | Batch size huan luyen |
| `lr` | `1e-4` | Learning rate |
| `hidden_size` | `128` | Kich thuoc hidden representation va projection dim |
| `lambda_con` | `0.1` | Trong so cua SupCon trong loss ket hop |
| `temperature` | `0.07` | Temperature tau cua SupCon |
| `mask_ratio` | `0.15` | Ti le mask theo thoi gian va feature cho augmentation |
| `noise_std` | `0.05` | Do lech chuan Gaussian noise cho augmentation |
| `num_layers` | `1` | So layer trong encoder |
| `cls_layers` | `1` | So hidden layer trong classifier head |
| `num_workers` | `0` | DataLoader workers |
| `optimizer` | Adam | `torch.optim.Adam(model.parameters(), lr=lr)` |
| `monitor` | Val AUC | Luu checkpoint tot nhat theo validation AUC; neu bang AUC thi so F1 |

Du lieu dau vao sau khi load co dang:

```text
X: (N, 5, 7, 22)
y: (N,)
```

Trong loader, `X` duoc flatten thanh:

```text
seq_feat: (B, 5 * 7 * 22)
```

Trong `SiameseLGB._preprocess`, tensor duoc reshape thanh:

```text
x: (B, 35, 22)
```

Trong do:

- `B`: batch size.
- `35 = 5 weeks * 7 days`.
- `22`: so activity features.

## 3. Kien Truc Forward

### 3.1 BCE-only sau khi fix

`bce_only` khong con chay hai nhanh Siamese. Code goi:

```python
logits = model.forward_single(batch)
loss = BCEWithLogitsLoss(logits, y)
```

Duong forward:

```text
x -> preprocess -> encoder -> h -> classifier -> logits
```

Dac diem:

- Khong goi `self.augment`.
- Khong tao `view1`, `view2`.
- Khong chay projection head.
- Khong tao `z1`, `z2`.
- Model van o `train()` mode, nen Dropout trong classifier van hoat dong dung che do training.

Day la baseline BCE sach:

```text
L = L_BCE
```

### 3.2 SupCon-only

`supcon_only` chay full Siamese training forward:

```text
x -> augment -> view1, view2
view1 -> shared encoder -> h1 -> projection head -> z1
view2 -> shared encoder -> h2 -> projection head -> z2
classifier(h1) -> logits
```

Nhung loss chi dung `z1`, `z2`, `y`:

```text
L = L_SupCon
```

`logits` van duoc tinh ra trong forward, nhung khong dong gop vao loss.

### 3.3 DP-SCL / Combined

`combined` cung chay full Siamese training forward:

```text
x -> augment -> view1, view2
view1 -> encoder -> h1 -> projection head -> z1
view2 -> encoder -> h2 -> projection head -> z2
h1 -> classifier -> logits
```

Tong loss:

```text
L_DP-SCL = L_BCE + lambda * L_SupCon
```

Voi cau hinh hien tai:

```text
lambda = 0.1
tau = 0.07
```

## 4. L_BCE

`L_BCE` duoc cai dat bang:

```python
torch.nn.BCEWithLogitsLoss()
```

Input:

```text
logits: (B, 1)
y:      (B, 1), label nhi phan 0/1
```

Cong thuc cho moi mau `i`:

```text
L_BCE_i = - y_i * log(sigmoid(s_i))
          - (1 - y_i) * log(1 - sigmoid(s_i))
```

Trong do:

- `s_i` la logit dau ra cua classifier.
- `sigmoid(s_i)` la xac suat du doan lop positive.

Loss batch:

```text
L_BCE = mean_i L_BCE_i
```

Trong code, `BCEWithLogitsLoss` tinh sigmoid ben trong de on dinh so hoc hon so voi viec goi sigmoid rieng.

## 5. L_SupCon

`L_SupCon` duoc cai dat trong `SupConLoss`.

Input trong loss ablation:

```python
features = torch.stack([z1, z2], dim=1)
supcon_loss = supcon(features, y.view(-1))
```

Dang tensor:

```text
z1:       (B, D)
z2:       (B, D)
features: (B, 2, D)
y:        (B,)
```

Trong `SupConLoss`, neu input la `(B, n_views, D)`, code flatten thanh:

```text
features -> (B * n_views, D)
labels   -> repeat_interleave(labels, n_views)
```

Voi hai view:

```text
features: (2B, D)
labels:   (2B,)
```

Projection `z1`, `z2` la embedding da L2-normalized tu projection head.

### 5.1 Similarity

Voi moi cap embedding `i, j`:

```text
sim(i, j) = z_i^T z_j / tau
```

Trong do:

- `tau` la temperature, hien tai `0.07`.
- Vi `z` da normalize, `z_i^T z_j` tuong duong cosine similarity.

### 5.2 Positive va negative pairs

Code tao mask:

```text
mask[i, j] = 1 neu label_i == label_j
mask[i, i] = 0
```

Nghia la:

- Positive pairs: cac embedding trong batch co cung label.
- Negative pairs: cac embedding khac label.
- Self-pair bi loai bo.

Neu toan batch chi co mot label duy nhat, code tra ve loss `0.0` de tranh truong hop khong co negative pair.

### 5.3 Cong thuc SupCon

Voi anchor `i`, tap positive la `P(i)`:

```text
L_i = - 1 / |P(i)| * sum_{p in P(i)}
      log exp(sim(i, p)) / sum_{a != i} exp(sim(i, a))
```

Loss batch:

```text
L_SupCon = mean_i L_i
```

Trong code co them buoc tru max logits truoc khi exp de on dinh so hoc:

```text
logits = similarity - max(similarity)
```

## 6. Tong Loss cua DP-SCL

Trong mode `combined`, code tinh:

```python
bce_loss = bce(logits, y_batch)
supcon_loss = supcon(features, y_batch.view(-1))
loss = bce_loss + lambda_con * supcon_loss
```

Cong thuc:

```text
L_total = L_BCE + lambda * L_SupCon
```

Voi cau hinh hien tai:

```text
L_total = L_BCE + 0.1 * L_SupCon
```

Y nghia:

- `L_BCE` toi uu truc tiep kha nang phan loai dropout/non-dropout.
- `L_SupCon` ep representation cua cac mau cung label gan nhau va khac label xa nhau.
- `lambda` dieu khien muc anh huong cua representation learning len tong loss.

## 7. Augmentation trong DP-SCL

Augmentation chi duoc dung trong cac mode hai nhanh (`supcon_only`, `combined`).

Moi view duoc tao bang:

```text
view = time_mask(feature_mask(x)) + GaussianNoise(0, sigma^2)
```

Chi tiet:

- Time masking: mask ngau nhien cac timestep/ngay.
- Feature masking: mask ngau nhien cac activity features.
- Gaussian noise: cong nhieu voi `sigma = 0.05`.

Voi cau hinh hien tai:

```text
pt = pf = 0.15
sigma = 0.05
```

Hai view `view1`, `view2` duoc augment doc lap.

## 8. Bang Dien Giai Ba Cau Hinh

| Cau hinh | Forward | Augmentation | Projection | Classifier | Loss dung de update |
|---|---|---:|---:|---:|---|
| `bce_only` | mot nhanh | Khong | Khong | Co | `L_BCE` |
| `supcon_only` | hai nhanh | Co | Co | Co, nhung logits khong dung trong loss | `L_SupCon` |
| `combined` | hai nhanh | Co | Co | Co | `L_BCE + lambda * L_SupCon` |

## 9. Early Stopping va Danh Gia

Sau moi epoch:

1. Danh gia tren validation set.
2. Chon threshold theo F1 tren validation.
3. Tinh validation AUC va validation F1.
4. Luu checkpoint neu:

```text
val_auc tang
hoac val_auc bang nhau nhung val_f1 tang
```

5. Dung som neu khong cai thien trong `patience = 30` epoch.

Sau training:

1. Load checkpoint tot nhat.
2. Chon threshold tren validation set.
3. Danh gia tren test set.
4. Ghi `per_seed_results.csv`, `summary_results.csv`, va `report.txt`.

## 10. Ghi Chu Quan Trong

Sau khi sua code, `bce_only` khong con la Siamese forward bi cat loss. No la BCE baseline dung nghia:

```text
x -> encoder -> classifier -> BCE
```

Do do so sanh voi DP-SCL co y nghia hon:

```text
BCE only:     do nang luc classification cua encoder mot nhanh.
DP-SCL:   do them tac dong cua two-view augmentation va supervised contrastive learning.
SupCon only:  do kha nang representation learning khi khong co tin hieu BCE truc tiep.
```
