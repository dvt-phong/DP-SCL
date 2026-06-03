# Run Scripts

Run all commands from the project root.

## DP-SCL

```bash
python run_scripts/run_dp_scl_mode.py --seeds 42 --max-epochs 15
python run_scripts/run_dp_scl.py --seeds 42 --max-epochs 15
```

## Sensitivity

```bash
python run_scripts/run_lambda_sens.py --seeds 42 --max-epochs 15
python run_scripts/run_tau_sens.py --seeds 42 --max-epochs 15
bash run_scripts/run_lambda_tau_sensitivity_5seed.sh --seeds 42 --force
```

## Loss Ablation

```bash
python run_scripts/run_dp_scl_loss_ablation.py --seeds 42 --max-epochs 15
./run_scripts/run_dp_scl_loss_ablation_fixed.sh --dry-run
```

## Baseline/Ablation Helpers

```bash
python run_scripts/run_dl_lstm.py --seeds 42 --max-epochs 15
python run_scripts/run_dl_lstm_mha.py --seeds 42 --max-epochs 15
python run_scripts/run_siamese_lstm_mha.py --seeds 42 --max-epochs 15
python run_scripts/run_siamese_lambda0.py --seeds 42 --max-epochs 15
```

The Python runners automatically change their working directory to the project
root so imports and relative output paths remain stable.
