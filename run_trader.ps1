python -m execution.live_trader `
  --checkpoint ./experiments/checkpoints/best_model.pt `
  --heads-dir ./experiments/heads_gold `
  --m15-checkpoint ./experiments/checkpoints/m15_warmstart/best_model.pt `
  --m15-st-threshold 0.000200 `
  --symbol XAUUSD `
  --magic 20260101 `
  --dry-run
