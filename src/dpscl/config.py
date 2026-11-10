"""Shared constants and model-parameter builders for DP-SCL experiments."""

SEED_LIST = [1, 11, 111, 1111, 11111]
MODEL_NAME = "DP-SCL"
METRIC_NAMES = ["auc", "acc", "precision", "recall", "f1"]


def make_dp_scl_param_dict(args, ds_config):
    return {
        "activity_num": ds_config["activity_num"],
        "sta_day": ds_config["sta_day"],
        "week_count": ds_config["week_count"],
        "select_count": ds_config["week_count"],
        "cnn_in_channels": ds_config["days_per_week"],
        "supcon_hidden_size": args.hidden_size,
        "supcon_proj_dim": args.hidden_size,
        "supcon_temperature": args.temperature,
        "supcon_mask_ratio": args.mask_ratio,
        "supcon_noise_std": args.noise_std,
        "supcon_attn_heads": 4,
        "supcon_cls_dropout": 0.3,
        "supcon_num_layers": args.num_layers,
        "supcon_cls_hidden_layers": args.cls_layers,
    }

