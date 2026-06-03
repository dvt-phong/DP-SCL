# Tài Liệu Kỹ Thuật: Phương Pháp DP-SCL

## 1. Tổng quan

DP-SCL trong project này là mô hình **Temporal Siamese Network kết hợp Supervised Contrastive Learning** để dự đoán nguy cơ dropout/non-dropout từ chuỗi hoạt động học tập theo thời gian.

Code chính:

- `src/models/siamese.py`: kiến trúc Siamese, encoder, projection head, classifier.
- `src/models/common.py`: `AugmentationModule`, `LearnableQueryPool`, `SupConLoss`.
- `train.py`: cách ghép `BCEWithLogitsLoss` và `SupConLoss` thành total loss.

Luồng tổng quát:

```text
Input activity sequence
        |
        v
Preprocess: (B, 5, 7, 22) -> (B, 35, 22)
        |
        v
Training only: Augmentation -> view1, view2
        |
        v
Shared Siamese Encoder
        |                         |
        v                         v
      h1                        h2
        |                         |
        v                         v
Projection Head              Projection Head
        |                         |
        v                         v
      z1                        z2
        |
        v
Classifier(h1) -> logit

Loss = BCE(logit, y) + lambda_con * SupCon(z1, z2, y)
```

Trong inference/evaluation, augmentation bị tắt và mô hình chỉ chạy một nhánh:

```text
x -> encoder -> h -> classifier -> logit -> sigmoid(logit)
```

## 2. Input và preprocessing

Mỗi sample biểu diễn lịch sử hoạt động học tập theo 5 tuần, mỗi tuần 7 ngày, mỗi ngày có 22 loại action:

```text
x_raw: (B, 5, 7, 22)
```

Trong `SiameseLGB._preprocess`, input được reshape thành chuỗi ngày:

```text
x: (B, T, F) = (B, 35, 22)
```

Trong đó:

- `B`: batch size.
- `T = week_count * days_per_week = 5 * 7 = 35`.
- `F = activity_num = 22`.

Một timestep tương ứng với một ngày, vector feature tại timestep đó là số liệu của 22 loại hoạt động.

Các module tùy chọn:

- `ActionWeightedInput`: học trọng số quan trọng cho từng loại action.
- `EarlyPredictionMask`: che các tuần sau để phục vụ bài toán dự đoán sớm.

## 3. Siamese Network

Siamese Network dùng **hai nhánh có cùng trọng số**. Hai nhánh nhận hai biến thể khác nhau của cùng một sample, nhưng đi qua cùng một encoder và cùng projection head.

Trong training:

```text
x -> AugmentationModule -> view1, view2
view1 -> encoder -> h1 -> projection_head -> z1
view2 -> encoder -> h2 -> projection_head -> z2
h1 -> classifier -> logit
```

Điểm quan trọng:

- Encoder của hai nhánh là cùng một object `self.encoder`, không phải hai encoder độc lập.
- Projection head cũng được share weight.
- Classifier chỉ dùng `h1`, không dùng `z1`.
- `z1`, `z2` chỉ phục vụ SupCon loss.

Ý nghĩa kỹ thuật:

- Hai view của cùng một sample nên có representation gần nhau.
- Các sample cùng label trong batch nên gần nhau trong contrastive space.
- Các sample khác label nên bị đẩy xa nhau.
- BCE vẫn giữ tín hiệu phân loại trực tiếp cho nhiệm vụ dropout prediction.

## 4. Augmentation Module

`AugmentationModule` tạo hai view độc lập của cùng input:

```text
view1 = augment(x)
view2 = augment(x)
```

Mỗi view dùng ba loại nhiễu:

| Thành phần | Mô tả | Mục đích |
|---|---|---|
| Time masking | Random zero-out một số timestep/ngày | Giúp model không phụ thuộc cứng vào một vài ngày cụ thể |
| Feature masking | Random zero-out một số action feature | Giúp model bền hơn khi một số loại hoạt động bị thiếu/nhiễu |
| Gaussian noise | Cộng nhiễu chuẩn `N(0, noise_std^2)` | Tăng tính ổn định của representation |

Mặc định trong code:

```text
time_mask_ratio = 0.15
feat_mask_ratio = 0.15
noise_std = 0.05
```

Augmentation chỉ dùng khi `model.training == True`. Khi evaluation/inference, model không tạo hai view.

## 5. Encoder: LSTM, MHA, LQ Pooling

Encoder nhận input:

```text
x: (B, 35, 22)
```

và trả về vector representation:

```text
h: (B, hidden_size)
```

Mặc định `hidden_size = 128`.

### 5.1 LSTM Encoder

Mode:

```text
siamese_lstm
```

Kiến trúc:

```text
(B, 35, 22) -> LSTM(input_size=22, hidden_size=H) -> last timestep -> h
```

Trong code, output dùng:

```text
h = rnn_out[:, -1, :]
```

LSTM học quan hệ tuần tự theo ngày. Vì bài toán dropout phụ thuộc vào diễn biến hành vi qua thời gian, LSTM giúp mô hình nhớ xu hướng và thay đổi của hoạt động học tập.

### 5.2 BiLSTM Encoder

Mode:

```text
siamese_bilstm
```

Kiến trúc:

```text
(B, 35, 22) -> BiLSTM(hidden_size=H/2 mỗi chiều)
             -> concat forward/backward hidden
             -> h: (B, H)
```

Trong code:

```text
h = concat(h_n[-2], h_n[-1])
```

BiLSTM đọc chuỗi theo cả hai chiều, phù hợp khi muốn khai thác quan hệ giữa các ngày trước/sau trong toàn bộ cửa sổ quan sát.

### 5.3 LSTM + Multi-Head Attention

Mode:

```text
siamese_lstm_mha
```

Kiến trúc:

```text
x -> LSTM -> rnn_out: (B, T, H)
  -> MultiHeadAttention(Q=K=V=rnn_out)
  -> residual + LayerNorm
  -> mean pooling
  -> h: (B, H)
```

Trong code:

```text
attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
attn_out = self.layer_norm(attn_out + rnn_out)
h = mean(attn_out, dim=1)
```

Vai trò của MHA:

- Cho phép mỗi timestep attend đến các timestep khác.
- Học quan hệ dài hạn tốt hơn so với chỉ lấy hidden cuối của LSTM.
- Residual connection giữ lại thông tin gốc từ LSTM.
- LayerNorm ổn định phân phối activation.

### 5.4 LSTM + MHA + Learnable Query Pooling

Mode DP-SCL chính thường dùng:

```text
siamese_lstm_attn
```

Kiến trúc:

```text
x -> LSTM -> rnn_out: (B, T, H)
  -> MultiHeadAttention
  -> residual + LayerNorm
  -> LearnableQueryPool
  -> h: (B, H)
```

Learnable Query Pooling dùng một query vector học được:

```text
q: (1, 1, H)
q expanded -> (B, 1, H)
context = Attention(q, K=sequence, V=sequence)
h = context.squeeze(1)
```

Khác với mean pooling, LQ Pooling không xem mọi timestep quan trọng như nhau. Query vector được học trong training để tự chọn các giai đoạn có thông tin nhất cho dự đoán dropout.

### 5.5 BiLSTM + MHA + Learnable Query Pooling

Mode:

```text
siamese_bilstm_attn
```

Kiến trúc tương tự `siamese_lstm_attn`, nhưng block recurrent là BiLSTM:

```text
x -> BiLSTM -> MHA -> residual + LayerNorm -> LearnableQueryPool -> h
```

BiLSTM cung cấp context hai chiều, MHA học quan hệ giữa các timestep, còn LQ Pooling nén chuỗi thành một vector đại diện.

## 6. Projection Head

Projection head nhận representation `h` và tạo embedding `z` cho contrastive learning:

```text
h: (B, H) -> ProjectionHead -> z: (B, proj_dim)
```

Kiến trúc trong `SiameseProjectionHead`:

```text
Linear(H -> H)
ReLU
Linear(H -> proj_dim)
L2 normalize
```

Mặc định:

```text
H = 128
proj_dim = 128
```

Công thức:

```text
z_raw = W2 * ReLU(W1 * h + b1) + b2
z = z_raw / ||z_raw||_2
```

Lý do cần projection head:

- Không gian `h` phục vụ classification.
- Không gian `z` phục vụ contrastive learning.
- Tách hai không gian giúp SupCon không ép trực tiếp toàn bộ representation phân loại.
- L2 normalize giúp dot product giữa hai vector tương đương cosine similarity, tránh việc độ lớn vector làm lệch similarity.

## 7. Classifier

Classifier nhận `h`, không nhận `z`:

```text
h1 -> Classifier -> logit
```

Kiến trúc mặc định:

```text
Linear(H -> 64)
ReLU
Dropout(0.3)
Linear(64 -> 1)
```

Nếu `siamese_cls_hidden_layers > 1`, classifier thêm các hidden layer nhỏ hơn:

```text
Linear(dim -> dim/2)
ReLU
Dropout
```

Output của classifier là **logit thô**:

```text
s: (B, 1)
```

Không sigmoid trước khi tính loss, vì training dùng `BCEWithLogitsLoss`, hàm này tự kết hợp sigmoid và BCE theo cách ổn định số học hơn.

Khi cần xác suất trong evaluation:

```text
p = sigmoid(s)
```

## 8. BCE Loss

BCE loss là loss phân loại nhị phân cho dropout/non-dropout:

```text
y ∈ {0, 1}
s = classifier(h1)
p = sigmoid(s)
```

Công thức theo từng sample:

```text
L_BCE_i = - y_i * log(p_i) - (1 - y_i) * log(1 - p_i)
```

Loss trên batch:

```text
L_BCE = mean_i L_BCE_i
```

Trong code, BCE được tính bằng:

```text
binary_cross_entropy_with_logits(logits, targets)
```

hoặc biến thể weighted/focal nếu cấu hình sampling yêu cầu. Với cấu hình DP-SCL chuẩn, thành phần BCE là tín hiệu trực tiếp để classifier học ranh giới dropout/non-dropout.

## 9. Supervised Contrastive Loss

SupCon loss dùng `z1`, `z2` và label `y`.

Trong training:

```text
features = stack([z1, z2], dim=1)
features shape = (B, 2, proj_dim)
```

Trong `SupConLoss`, tensor được flatten:

```text
(B, 2, D) -> (2B, D)
labels: (B,) -> repeat_interleave -> (2B,)
```

Similarity giữa hai embedding:

```text
sim(i, j) = z_i^T z_j / tau
```

Trong đó:

- `z_i`, `z_j` đã L2-normalized.
- `tau` là temperature, mặc định `0.07`.

Positive pair của anchor `i` là các vector có cùng label:

```text
P(i) = {j | y_j = y_i, j != i}
```

Loss cho anchor `i`:

```text
L_i = - 1 / |P(i)| * sum_{p in P(i)}
      log( exp(sim(i, p)) / sum_{a != i} exp(sim(i, a)) )
```

Loss toàn batch:

```text
L_SupCon = mean_i L_i
```

Ý nghĩa:

- Nếu hai sample cùng label, SupCon kéo embedding của chúng lại gần nhau.
- Nếu hai sample khác label, SupCon đẩy chúng ra xa nhau qua mẫu số softmax.
- Hai view của cùng một sample luôn có cùng label nên cũng là positive pair.

Trong code có guard:

- Nếu batch size quá nhỏ, trả loss 0.
- Nếu toàn batch chỉ có một label, trả loss 0 vì không có negative pair.

## 10. Total Loss của DP-SCL

Total loss dùng để update mô hình:

```text
L_total = L_BCE + lambda_con * L_SupCon
```

Mặc định:

```text
lambda_con = 0.1
temperature = 0.07
```

Trong `train.py`:

```text
pred, z1, z2 = model(sub_graph)
bce_loss = bce_loss_fn(pred, ground_truth)
features = torch.stack([z1, z2], dim=1)
con_loss = supcon_criterion(features, ground_truth.view(-1))
loss = bce_loss + lambda_con * con_loss
```

Vai trò từng thành phần:

| Thành phần | Vai trò |
|---|---|
| `L_BCE` | Tối ưu trực tiếp khả năng phân loại dropout/non-dropout |
| `L_SupCon` | Tổ chức representation theo label trong không gian projection |
| `lambda_con` | Điều khiển mức ảnh hưởng của SupCon so với BCE |
| `temperature` | Điều chỉnh độ sắc của contrastive softmax |

Nếu `lambda_con = 0`, mô hình trở thành Siamese-style encoder nhưng chỉ tối ưu bằng BCE:

```text
L_total = L_BCE
```

Nếu bỏ BCE và chỉ dùng SupCon, representation có thể học cụm tốt nhưng classifier không nhận tín hiệu phân loại trực tiếp.

## 11. Training và inference

### Training

```text
model.train()
x -> augmentation -> view1, view2
view1 -> encoder -> h1 -> projection -> z1
view2 -> encoder -> h2 -> projection -> z2
h1 -> classifier -> logit
loss = BCE(logit, y) + lambda_con * SupCon(z1, z2, y)
backpropagation
optimizer.step()
```

Training forward trả:

```text
(logits, z1, z2)
```

### Inference/Evaluation

```text
model.eval()
x -> encoder -> h -> classifier -> logit
probability = sigmoid(logit)
```

Inference forward chỉ trả:

```text
logits
```

Điều này giúp inference nhẹ hơn training vì không cần augmentation, không cần chạy hai nhánh, và không cần projection head cho SupCon.

## 12. Bảng tóm tắt thành phần

| Thành phần | Input | Output | Dùng khi | Chức năng |
|---|---:|---:|---|---|
| Preprocess | `(B, 5, 7, 22)` | `(B, 35, 22)` | Train + eval | Chuyển dữ liệu tuần/ngày/action thành chuỗi thời gian |
| Augmentation | `(B, 35, 22)` | `view1`, `view2` | Train | Tạo hai view cho Siamese/SupCon |
| Shared Encoder | `(B, 35, 22)` | `h: (B, H)` | Train + eval | Học representation chuỗi hoạt động |
| MHA | `(B, T, H)` | `(B, T, H)` | Mode attention | Học quan hệ giữa các timestep |
| LQ Pooling | `(B, T, H)` | `(B, H)` | `*_attn` | Nén chuỗi bằng query học được |
| Projection Head | `h: (B, H)` | `z: (B, D)` | Train | Tạo embedding cho SupCon |
| Classifier | `h: (B, H)` | `logit: (B, 1)` | Train + eval | Dự đoán dropout/non-dropout |
| BCE Loss | `logit`, `y` | scalar | Train | Loss phân loại |
| SupCon Loss | `z1`, `z2`, `y` | scalar | Train | Loss contrastive có giám sát |
| Total Loss | `L_BCE`, `L_SupCon` | scalar | Train | Loss cuối để backprop |

## 13. Cấu hình DP-SCL đề xuất trong project

Mode đại diện cho DP-SCL đầy đủ:

```text
siamese_lstm_attn
```

Tương ứng:

```text
LSTM -> Multi-Head Attention -> Learnable Query Pooling
```

Các hyperparameter chính:

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---:|---|
| `siamese_hidden_size` | `128` | Kích thước representation `h` |
| `siamese_proj_dim` | `128` | Kích thước projection `z` |
| `siamese_temperature` | `0.07` | Temperature của SupCon |
| `lambda_con` | `0.1` | Trọng số SupCon trong total loss |
| `siamese_mask_ratio` | `0.15` | Tỷ lệ masking time/feature |
| `siamese_noise_std` | `0.05` | Độ lệch chuẩn Gaussian noise |
| `siamese_attn_heads` | `4` | Số head của MHA trong encoder |
| `siamese_cls_dropout` | `0.3` | Dropout của classifier |

## 14. Kết luận kỹ thuật

DP-SCL kết hợp hai mục tiêu học:

```text
Classification objective:
    h -> classifier -> dropout logit -> BCE

Representation objective:
    h -> projection head -> z -> SupCon
```

Nhờ Siamese augmentation, encoder học representation ổn định trước nhiễu. Nhờ SupCon, representation được tổ chức theo label. Nhờ BCE, classifier vẫn được tối ưu trực tiếp cho bài toán dự đoán dropout. Thành phần LSTM/MHA/LQ Pooling giúp encoder vừa nắm được xu hướng tuần tự, vừa học được quan hệ giữa các ngày, vừa chọn các timestep quan trọng nhất để tạo vector đại diện cuối cùng.
