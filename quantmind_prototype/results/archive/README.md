# Archived artifacts

These files predate the day-alignment fix (see `src/backtest.align_to_tradeable`)
and the controlled W3 retraining protocol. They are kept for provenance only —
**nothing in the report or the current pipeline reads from this folder.**

- `run_summary.json`, produced by the old `run_prototype.py` (single run, pre-alignment)
- `multiseed.json` / `multiseed.csv`, produced by the old `run_experiments_train.py`
  (5 price-only seeds, pre-alignment 1/N baseline)
- `full_run.log`, a console log from the same pre-alignment run (1/N Sharpe 0.8153,
  total return 42.0%; the current, cited figures are 0.81 and 41.6%)

The current, cited numbers live in `results/evaluation.json`, `results/eval_main_table.csv`
and `results/eval_stats.json`, produced by `run_experiments_w3.py` + `run_evaluation.py`.
