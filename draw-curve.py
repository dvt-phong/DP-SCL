import json, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Định nghĩa màu chữ tập trung ──────────────────
FONT_COLOR = "black"   # đổi 1 chỗ này là thay đổi toàn bộ
FONT_SIZE  = 13

# Load data
with open("./datastore/loss_curve/history_dp_scl.json") as f:
    dp = json.load(f)
with open("./datastore/loss_curve/history_wo_supcon.json") as f:
    wo = json.load(f)

# Smoothing function (rolling average)
def smooth(data, w=10):
    arr = np.array(data)
    result = np.convolve(arr, np.ones(w)/w, mode='valid')
    return list(range(w, 201)), result.tolist()

ep_s, dp_bce_s = smooth(dp["train_bce_loss"])
_,    wo_bce_s = smooth(wo["train_bce_loss"])
_,    dp_sup_s = smooth(dp["train_supcon_loss"])
_,    dp_auc_s = smooth(dp["val_auc"])
_,    wo_auc_s = smooth(wo["val_auc"])

# Best checkpoint epochs
epochs = list(range(1, 201))
dp_best_ep = epochs[dp["val_auc"].index(max(dp["val_auc"]))]  # 81
wo_best_ep = epochs[wo["val_auc"].index(max(wo["val_auc"]))]  # 69

BLUE   = "#2A4A7F"
ORANGE = "#F0A500"

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["(a) Train BCE Loss", "(b) Train SupCon Loss", "(c) Validation AUC"],
    horizontal_spacing=0.09
)

# (a) BCE Loss — cả 2 model
fig.add_trace(go.Scatter(x=ep_s, y=dp_bce_s, name="DP-SCL",
    mode="lines", line=dict(color=BLUE, width=2.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=ep_s, y=wo_bce_s, name="w/o SupCon",
    mode="lines", line=dict(color=ORANGE, width=2.2, dash="dash")), row=1, col=1)

# (b) SupCon Loss — chỉ DP-SCL
fig.add_trace(go.Scatter(x=ep_s, y=dp_sup_s,
    mode="lines", line=dict(color=BLUE, width=2.2), showlegend=False), row=1, col=2)

# (c) Validation AUC — cả 2 model
fig.add_trace(go.Scatter(x=ep_s, y=dp_auc_s,
    mode="lines", line=dict(color=BLUE, width=2.2), showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=ep_s, y=wo_auc_s,
    mode="lines", line=dict(color=ORANGE, width=2.2, dash="dash"), showlegend=False), row=1, col=3)

# Vertical dotted lines đánh dấu best checkpoint
ymin, ymax = 0.832, 0.863
for ep, color in [(dp_best_ep, BLUE), (wo_best_ep, ORANGE)]:
    fig.add_trace(go.Scatter(
        x=[ep, ep], y=[ymin, ymax],
        mode="lines",
        line=dict(color=color, dash="dot", width=1.5),
        showlegend=False), row=1, col=3)

fig.update_layout(
    font=dict(size=FONT_SIZE, color=FONT_COLOR),
    title=dict(
        font=dict(size=FONT_SIZE,color=FONT_COLOR), x=0.5, xanchor="center",
    ),
    legend=dict(orientation='h', yanchor='top', y=-0.18,
                xanchor='center', x=0.5, font=dict(size=FONT_SIZE, color=FONT_COLOR)),
    height=400, width=1150,
    margin=dict(t=80, b=100, l=65, r=30),
    plot_bgcolor="white", paper_bgcolor="white"
)

fig.update_annotations(
    font=dict(size=FONT_SIZE, color=FONT_COLOR),
    yshift=8
)

checkpoints = sorted([(dp_best_ep, BLUE), (wo_best_ep, ORANGE)])
label_y = ymin + 0.0012
for i, (ep, color) in enumerate(checkpoints):
    xanchor = "right" if i == 0 else "left"
    xshift = -6 if i == 0 else 6
    fig.add_annotation(
        x=ep, y=label_y,
        text=f"ep.{ep}",
        showarrow=False,
        xanchor=xanchor,
        yanchor="bottom",
        xshift=xshift,
        font=dict(size=FONT_SIZE, color=FONT_COLOR),
        row=1, col=3
    )

# Shared axis style
AXIS_STYLE = dict(
    showgrid=False,                          # tắt lưới bên trong
    showline=True, linecolor="black", linewidth=1, mirror=True,  # viền 4 cạnh
    zeroline=False,
    tickfont=dict(size=FONT_SIZE, color=FONT_COLOR),
    title_font=dict(size=FONT_SIZE, color=FONT_COLOR),  # axis title
)

for col in [1, 2, 3]:
    fig.update_xaxes(title_text="Epoch", dtick=50, **AXIS_STYLE, row=1, col=col)

fig.update_yaxes(title_text="BCE Loss",    **AXIS_STYLE, row=1, col=1)
fig.update_yaxes(title_text="SupCon Loss", **AXIS_STYLE, row=1, col=2)
fig.update_yaxes(title_text="Val AUC", range=[0.832, 0.864], **AXIS_STYLE, row=1, col=3)

fig.write_image("loss_curve_3subplots.png", scale=2)
