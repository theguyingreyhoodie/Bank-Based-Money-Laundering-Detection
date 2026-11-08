"""Flower ClientApp: one simulated bank.

Each simulated node is one client bank. Its `partition-id` (0..n-1) picks the folder
data/partitions/client_<id>/, and the bank only ever reads that folder.

mode "naive": the bank keeps only the core columns (shared by all banks) with the shared
category lists, so every bank has the same input layout, and the WHOLE BankModel is
exchanged and averaged.

mode "encoder_head" (our method): the bank uses its full own schema and its own category
lists. Only the SharedHead is exchanged and averaged. The encoder stays at the bank: it is
kept in the node's `context.state` between rounds and is never put in a reply message.

mode "hybrid": the bank's columns are laid out as [core | own]. A core encoder (on the core
columns, with the shared category lists) and the head are exchanged and averaged; the private
encoder on the bank's own columns stays in `context.state`.

mode "zero_filled" (baseline): all banks agree on one global column list (the union of their
columns) and lay out their data in it; columns a bank lacks are 0 with a 0/1 "<missing>" flag
(preprocess.load_client_zero_filled). The WHOLE BankModel is exchanged and averaged.

In every mode the rule is the same: a parameter whose name starts with PRIVATE_PREFIX[mode]
stays at the bank, everything else is sent to the server (see shared_state).

Each bank keeps the round whose model does best on its own validation split (PR-AUC), as the
MLP baselines keep their best epoch: whenever a round beats the bank's best so far, the bank
writes its test scores to <out_dir>/scores/ and that model to <out_dir>/models/ (for
explanations and alerts, see scripts/build_alerts.py). Both stay with the experimenter; nothing
of it is sent to the server.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ..features import SHARED_VOCAB
from ..profiles import PROFILE_VOCAB
from ..metrics import best_f1_threshold
from ..metrics import evaluate as eval_metrics
from ..models import BankModel, HybridModel, pos_weight, predict_proba, train_epochs
from ..preprocess import ClientData, load_client, load_client_hybrid, load_client_zero_filled, undersample
from ..utils import load_json, save_json
from .strategy import AUDIT_KEY, BANK_KEY, hash_parameters

log = logging.getLogger(__name__)

MODES = ("naive", "zero_filled", "encoder_head", "hybrid")
# Parameters whose names start with this prefix never leave the bank (None: everything is shared).
PRIVATE_PREFIX = {"naive": None, "zero_filled": None, "encoder_head": "encoder.", "hybrid": "private_encoder."}
# Category lists for every categorical column, for the zero-filled layout.
ALL_VOCAB = {**SHARED_VOCAB, **PROFILE_VOCAB}
ENCODER_KEY = "encoder"        # context.state: this bank's private encoder weights
PROGRESS_KEY = "progress"      # context.state: {"encoder_rounds": n}
BEST_KEY = "best"              # context.state: {"val_pr_auc": best so far, "round": the round it came from}


def new_model(mode: str, d: ClientData) -> torch.nn.Module:
    if mode == "hybrid":
        return HybridModel(d.core_dim, d.input_dim - d.core_dim)
    return BankModel(input_dim=d.input_dim)


def is_private(mode: str, key: str) -> bool:
    return PRIVATE_PREFIX[mode] is not None and key.startswith(PRIVATE_PREFIX[mode])


def load_bank_data(mode: str, path: str | Path, global_columns: list[str] | None = None) -> ClientData:
    """A bank's arrays in the layout the given mode trains on."""
    if mode == "hybrid":
        return load_client_hybrid(path, SHARED_VOCAB)
    if mode == "naive":
        return load_client(path, "core", SHARED_VOCAB)
    if mode == "zero_filled":
        return load_client_zero_filled(path, global_columns, ALL_VOCAB)
    return load_client(path)


# Ray sends the ClientApp to its worker processes anew with every message, so a cache kept inside
# make_client_app is empty again on the next message: each round used to reload and re-preprocess
# the bank's data twice (train and evaluate). Instead the first load writes the bank's arrays to
# the run's cache folder, and each worker process keeps them memory-mapped in this module-level
# dict, which lives as long as the process. Later rounds read them back almost for free, and all
# processes share one copy in memory.
_BANKS: dict[str, ClientData] = {}
_PARTS, _SPLITS = ("X", "y", "txn_ids"), ("train", "val", "test")


def _write_bank(d: ClientData, folder: Path) -> None:
    tmp = folder.with_name(f"{folder.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for part in _PARTS:
        for s in _SPLITS:
            np.save(tmp / f"{part}_{s}.npy", getattr(d, part)[s])
    save_json({"name": d.name, "feature_names": d.feature_names, "core_dim": d.core_dim}, tmp / "meta.json")
    try:
        tmp.rename(folder)            # readers see the folder complete or not at all
    except OSError:                   # another process wrote it first
        shutil.rmtree(tmp, ignore_errors=True)


def cached_bank_data(mode: str, path: str | Path, cache_dir: str | Path,
                     global_columns: list[str] | None = None) -> ClientData:
    """load_bank_data, computed once per run and memory-mapped after that (read-only arrays)."""
    folder = Path(cache_dir) / f"{mode}-{Path(path).name}"
    key = str(folder)
    if key not in _BANKS:
        if not (folder / "meta.json").exists():
            _write_bank(load_bank_data(mode, path, global_columns), folder)
        meta = load_json(folder / "meta.json")
        arrays = {p: {s: np.load(folder / f"{p}_{s}.npy", mmap_mode="r") for s in _SPLITS} for p in _PARTS}
        _BANKS[key] = ClientData(meta["name"], arrays["X"], arrays["y"], arrays["txn_ids"], meta["feature_names"],
                                 [], meta["core_dim"])
    return _BANKS[key]


def shared_state(model: torch.nn.Module, mode: str) -> dict[str, torch.Tensor]:
    """The parameters this mode sends to the server."""
    return {k: v for k, v in model.state_dict().items() if not is_private(mode, k)}


def private_state(model: torch.nn.Module, mode: str) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if is_private(mode, k)}


def make_client_app(partitions_dir: str | Path, out_dir: str | Path, mode: str = "naive",
                    lr: float = 1e-3, batch_size: int = 512, local_epochs: int = 1,
                    neg_per_pos: int = 20, seed: int = 42,
                    global_columns: list[str] | None = None) -> ClientApp:
    """global_columns: the agreed column list (zero_filled mode only). The banks' preprocessed
    arrays are cached in <out_dir>/cache/ (run_fl.py deletes it after the run)."""
    if mode not in MODES:
        raise ValueError(f"Unknown FL mode {mode!r}; expected one of {MODES}")
    if mode == "zero_filled" and not global_columns:
        raise ValueError("zero_filled mode needs global_columns")
    has_private = PRIVATE_PREFIX[mode] is not None
    app = ClientApp()

    def bank(context: Context) -> tuple[int, ClientData]:
        pid = int(context.node_config["partition-id"])
        path = Path(partitions_dir) / f"client_{pid}"
        return pid, cached_bank_data(mode, path, Path(out_dir) / "cache", global_columns)

    def build_model(msg: Message, context: Context, pid: int, d: ClientData) -> torch.nn.Module:
        """Global weights from the server, plus this bank's own private encoder (if any)."""
        torch.set_num_threads(1)  # simulated banks run in parallel processes
        rnd = int(msg.content["config"]["server-round"])
        torch.manual_seed(seed + 1000 * pid + rnd)  # reproducible init and shuffling per bank/round
        model = new_model(mode, d)
        received = msg.content["arrays"].to_torch_state_dict()
        if set(received) != set(shared_state(model, mode)):
            raise ValueError(f"Server sent {sorted(received)}, expected the {mode} shared parameters")
        state = model.state_dict()
        state.update(received)
        if ENCODER_KEY in context.state:  # first round: freshly initialised encoder
            state.update(context.state[ENCODER_KEY].to_torch_state_dict())
        model.load_state_dict(state)
        return model

    def encoder_rounds(context: Context) -> int:
        return int(context.state[PROGRESS_KEY]["encoder_rounds"]) if PROGRESS_KEY in context.state else 0

    def save_round(model: torch.nn.Module, d: ClientData, pid: int, rnd: int, thr: float, val_pr_auc: float) -> None:
        """This bank's test scores and model for round `rnd` (it beat the bank's earlier rounds)."""
        scores, models = Path(out_dir) / "scores", Path(out_dir) / "models"
        scores.mkdir(parents=True, exist_ok=True)
        models.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"txn_id": np.asarray(d.txn_ids["test"]), "label": d.y["test"].astype(int),
                      "score": predict_proba(model, d.X["test"]), "threshold": thr, "round": rnd}
                     ).to_parquet(scores / f"client_{pid}.parquet", index=False)
        torch.save(model.state_dict(), models / f"client_{pid}.pt")
        save_json({"mode": mode, "round": rnd, "val_pr_auc": val_pr_auc, "input_dim": d.input_dim,
                   "core_dim": d.core_dim, "threshold": thr, "feature_names": d.feature_names,
                   "global_columns": global_columns}, models / f"client_{pid}.json")

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        pid, d = bank(context)
        model = build_model(msg, context, pid, d)
        rnd = int(msg.content["config"]["server-round"])
        X, y = undersample(d.X["train"], d.y["train"], neg_per_pos=neg_per_pos, seed=seed + rnd)
        loss = train_epochs(model, X, y, epochs=local_epochs, lr=lr, batch_size=batch_size,
                            weight=pos_weight(y))
        metrics = {"train_loss": float(loss), "num-examples": len(d.y["train"])}
        shared = ArrayRecord(shared_state(model, mode))            # the ONLY thing that leaves the bank
        if has_private:
            context.state[ENCODER_KEY] = ArrayRecord(private_state(model, mode))
            context.state[PROGRESS_KEY] = ConfigRecord({"encoder_rounds": encoder_rounds(context) + 1})
            metrics["encoder_rounds"] = encoder_rounds(context)   # should equal the round number
        # The bank declares the hash of what it sends; the server checks it and logs it for the audit.
        audit = ConfigRecord({"bank": d.name, "update-hash": hash_parameters(shared)})
        return Message(content=RecordDict({"arrays": shared, "metrics": MetricRecord(metrics), AUDIT_KEY: audit}),
                       reply_to=msg)

    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        pid, d = bank(context)
        model = build_model(msg, context, pid, d)
        rnd = int(msg.content["config"]["server-round"])
        val = predict_proba(model, d.X["val"])
        has_pos = d.y["val"].sum() > 0
        thr = best_f1_threshold(d.y["val"], val) if has_pos else 0.5
        m = eval_metrics(d.y["val"], val, thr)
        val_pr_auc = float(np.nan_to_num(m["pr_auc"]))  # 0 when the bank has no val positives
        best = context.state[BEST_KEY] if BEST_KEY in context.state else None
        # Keep the best round on validation. With no validation positives there is nothing to
        # choose by, so the latest round is kept.
        if best is None or not has_pos or val_pr_auc > float(best["val_pr_auc"]):
            context.state[BEST_KEY] = ConfigRecord({"val_pr_auc": val_pr_auc, "round": rnd})
            save_round(model, d, pid, rnd, thr, val_pr_auc)
        metrics = MetricRecord({
            "val_pr_auc": val_pr_auc,
            "val_f1": m["f1"],
            "val_positives": m["positives"],
            "num-examples": len(d.y["val"]),
        })
        return Message(content=RecordDict({"metrics": metrics, BANK_KEY: ConfigRecord({"bank": d.name})}),
                       reply_to=msg)

    return app
