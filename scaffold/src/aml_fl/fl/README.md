# Federated learning (Phase 3)

Five client banks train one laundering detector with Flower, without sharing raw data.
Each simulated node is one bank and reads only `data/partitions/client_<partition-id>/`.
`schema.json` lists the client's columns and the `core_columns` that every client has.

Run with `python scripts/run_fl.py --mode <mode>` (see the main README).

## Setups

| Mode | Shared (FedAvg'd) | Private (stays at the bank) | Features each bank uses |
|---|---|---|---|
| `naive` (baseline) | whole `BankModel` | nothing | the 9 `core_columns` only |
| `zero_filled` (baseline) | whole `BankModel` | nothing | the agreed union of all banks' columns; missing ones are 0 plus a 0/1 "missing" flag |
| `encoder_head` | `SharedHead` | `Encoder` | the bank's full own schema |
| `hybrid` | core encoder + `SharedHead` | encoder of the bank's own columns | `[core | own]` |

`hybrid` (`models.HybridModel`) splits the input in two. The core columns use the public
category lists in `features.SHARED_VOCAB`, so their layout is identical at every bank and a
core encoder can be averaged. The bank's own history and profile columns go through a private
encoder. Both 16-dim outputs are concatenated and fed to the shared head. The bank's own column
list and that encoder never leave the bank.

One rule decides what is sent in every mode: a parameter whose name starts with
`client_app.PRIVATE_PREFIX[mode]` stays at the bank, everything else is sent
(`client_app.shared_state`). The private part is kept in the node's `context.state` between
rounds. Each bank reports `encoder_rounds`, which must equal the round number in
`history.csv`; a lower value means the private encoder was lost between rounds.

`zero_filled` (`preprocess.load_client_zero_filled`) needs every bank to agree on one column
list (`run_fl.agreed_columns`: the union of the banks' schemas) and on the category lists of
every categorical column (`client_app.ALL_VOCAB`). Agreeing on that list reveals which columns
each bank stores; the hybrid mode avoids this.

The key comparison is ours versus zero-filled FedAvg. If zero-filled scores higher, report the
gap as the accuracy cost of keeping the schema private. Every setup should land between the
local-only and centralized MLP results in `experiments/results/baselines.csv`.

## Results (HI-Small, 100 rounds, seeds 42-46)

Setup since 2026-11-09: each bank's receiver features count only its own customers' payments,
the synthetic profiles are pure noise (`label_signal` 0), each bank keeps its best validation
round, and updates are averaged in bank order. Results from before that (receiver features
counted over every bank, label signal 0.1, last round, 3 seeds: hybrid 0.139) are in the git
history and are not comparable.

Pooled test PR-AUC, mean ± std over 5 seeds (`scripts/summarize_seeds.py`):

| Setup | Pooled test PR-AUC | Per seed | F1 | FPR |
|---|---|---|---|---|
| Centralized XGBoost (pooled data, upper bound) | 0.359 ± 0.005 | | 0.382 | 0.0004 |
| Centralized MLP (pooled data, upper bound) | 0.259 ± 0.012 | | 0.348 | 0.0004 |
| **FL hybrid** | **0.242 ± 0.019** | 0.209 / 0.240 / 0.249 / 0.256 / 0.254 | 0.258 | 0.0010 |
| FL encoder_head | 0.200 ± 0.005 | 0.200 / 0.194 / 0.200 / 0.197 / 0.209 | 0.266 | 0.0007 |
| Local-only XGBoost | 0.194 ± 0.022 | | 0.269 | 0.0006 |
| FL zero_filled | 0.187 ± 0.017 | 0.189 / 0.195 / 0.157 / 0.200 / 0.194 | 0.234 | 0.0012 |
| Local-only MLP (lower bound) | 0.179 ± 0.009 | | 0.257 | 0.0008 |
| FL naive | 0.052 ± 0.004 | 0.047 / 0.050 / 0.057 / 0.054 / 0.050 | 0.088 | 0.0044 |

Paired by seed (`seeds_paired.csv`: both setups ran on the same data with the same seed):

| Comparison | A − B, mean [95% CI] | A better on | p (paired t-test) |
|---|---|---|---|
| hybrid vs zero_filled | +0.055 [+0.022, +0.087] | 5 of 5 seeds | 0.0095 |
| hybrid vs encoder_head | +0.041 [+0.018, +0.065] | 5 of 5 seeds | 0.0083 |
| hybrid vs local-only XGBoost | +0.047 [+0.036, +0.059] | 5 of 5 seeds | 0.0003 |
| hybrid vs local-only MLP | +0.063 [+0.038, +0.088] | 5 of 5 seeds | 0.0022 |
| centralized MLP vs hybrid | +0.018 [−0.006, +0.042] | 4 of 5 seeds | 0.11 |

- Hybrid beats every other setup that keeps the data at the banks, on every seed: zero-filled
  FedAvg, sharing only the head, and each bank training alone with an MLP or with XGBoost. It
  does so while keeping each bank's column list private, which zero-filled FedAvg cannot.
- The gap to centralized training of the same MLP (+0.018) is not significant over 5 seeds.
  Centralized XGBoost stays well ahead (0.359).
- Per bank (`docs/figures/fig4_per_bank`), 4 of 5 banks gain over training alone with the MLP;
  Acacia Bank (client_1) is level (−0.005). Against XGBoost alone, hybrid is ahead at 3 banks;
  XGBoost alone stays better at client_0 and client_4.
- The round each bank keeps (`best_round` in `fl_<mode>.csv`): hybrid median 93 (50 to 100);
  encoder_head median 40, as sharing only the head overfits in later rounds. 21% of all bank-runs
  kept round 100. Hybrid's validation PR-AUC is still rising at round 100
  (`docs/figures/fig2_training_curves`), so more rounds may help.
- Recall per pattern: fan-in is where pooled data clearly helps (centralized XGBoost 0.49,
  hybrid 0.31), as a bank sees only its own customers' share of a receiver's payers. Hybrid
  matches or beats centralized XGBoost on cycle, bipartite and random. 40% of test laundering
  belongs to no listed pattern, and every setup catches only 4-6% of it.
- Naive FedAvg (common columns only) flags 4x more legitimate transactions (FPR 0.0044), so its
  per-pattern recall is not comparable with the other setups.

### Why counting only the bank's own payments helps

The results before 2026-11-09 counted every bank's payments in the receiver features and scored
lower (hybrid 0.171 with pure-noise profiles), although that used more data. In this dataset,
laundering goes to receivers that the sending bank's own customers have rarely paid. On the test
rows of banks 1-4, a low count ranks laundering above legitimate payments with ROC-AUC 0.68 for
fan-in, 0.69 for the 24-hour burst and 0.75 for received volume when the bank counts its own
customers' payments, against 0.52, 0.56 and 0.63 when every bank's payments are counted. The
pooled count washes out how unfamiliar a receiver is to this bank. client_0, whose schema has no
receiver features, scores exactly as before when it trains alone. This is synthetic data (IBM
AMLworld); whether real banks see the same is open.

### Sensitivity: profiles with a label signal (`configs/sensitivity_signal0.1.yaml`)

In this check, some profile attributes of accounts that laundered in the training period are drawn
riskier (`label_signal` 0.1), seeds 42-44:

| Setup | Main runs (signal 0, 5 seeds) | Signal 0.1 (3 seeds) |
|---|---|---|
| Centralized XGBoost | 0.359 ± 0.005 | 0.340 ± 0.007 |
| Centralized MLP | 0.259 ± 0.012 | 0.241 ± 0.008 |
| **FL hybrid** | **0.242 ± 0.019** | **0.214 ± 0.014** |
| Local-only XGBoost | 0.194 ± 0.022 | 0.187 ± 0.016 |
| FL zero_filled | 0.187 ± 0.017 | 0.167 ± 0.012 |
| Local-only MLP | 0.179 ± 0.009 | 0.159 ± 0.015 |

- The label-derived cue lowers every setup, and their order stays the same
  (`docs/figures/fig5_label_signal`). It marks only accounts that laundered in the training
  period, so models learn something that does not carry over to new launderers in the test period.
- Hybrid is ahead of zero-filled FedAvg, local-only MLP and local-only XGBoost on all 3 seeds here
  too, but with 3 seeds none of these gaps is significant at 0.05 (p = 0.06, 0.06, 0.12).

## Training

- FedAvg (`strategy.AuditedFedAvg`), all 5 banks every round, weighted by training rows.
  100 rounds by default (`fl.num_rounds`). Replies are averaged in a fixed order (sorted by
  bank), not in arrival order, so runs are bit-for-bit reproducible: rerunning a seed gives the
  same global model hash in every round and the same test scores.
- Local training: each round, undersample the bank's training data again
  (`seed + round`), then `fl.local_epochs` epochs with `pos_weight` in the loss. `lr` and
  `batch_size` come from the `mlp:` section so the results are comparable with the baselines.
- `AuditedFedAvg` stores the SHA-256 of every aggregated model and of every bank's update in
  `audit.json`, for the blockchain audit log (Phase 4). Each bank declares its update's hash in
  its reply; the server refuses the round if the update it received hashes differently.

## Evaluation

- Each bank keeps the round whose model does best on its own validation split (PR-AUC), as the
  MLP baselines keep their best epoch, and tunes its threshold on that split
  (`best_f1_threshold`). It writes that round's test scores to `<out_dir>/scores/` and its model
  to `<out_dir>/models/` (for the experimenter only; nothing is sent to the server).
  `best_round` in `fl_<mode>.csv` shows which round each bank kept.
- `run_fl.py` reports per-client results and one pooled result (`metrics.pooled_evaluate`:
  every row flagged by its own bank's threshold), plus recall per pattern.

## Files

- `client_app.py`: Flower `ClientApp` (train / evaluate) for all modes.
- `server_app.py`: Flower `ServerApp`; writes `history.csv` and `audit.json`.
- `strategy.py`: `AuditedFedAvg` and `hash_parameters`.

## Speed

Ray sends the `ClientApp` to its worker processes anew with every message, so nothing cached
inside `make_client_app` survives to the next round. `client_app.cached_bank_data` therefore
writes each bank's preprocessed arrays to `<out_dir>/cache/` once and memory-maps them in every
worker process; `run_fl.py` deletes the folder after the run. A round after the first takes
under a second.

## Known limitations

- Each bank scales its own features, which matches the privacy story. In shared parts of the
  model, the same feature can therefore be scaled differently at different banks.
- If xgboost and torch run in the same process on macOS, set `torch.set_num_threads(1)`
  (see `scripts/run_baselines.py`).
