import json, csv

with open("./datastore/loss_curve/history_dp_scl.json") as f:
    dp = json.load(f)
with open("./datastore/loss_curve/history_wo_supcon.json") as f:
    wo = json.load(f)

with open("training_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch","BCE_DPSCL","BCE_wo","SupCon_DPSCL","ValAUC_DPSCL","ValAUC_wo"])
    for i in range(200):
        writer.writerow([
            dp["epoch"][i],
            dp["train_bce_loss"][i],
            wo["train_bce_loss"][i],
            dp["train_supcon_loss"][i],
            dp["val_auc"][i],
            wo["val_auc"][i]
        ])