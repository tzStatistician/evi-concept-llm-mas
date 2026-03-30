#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# Determinism-related env vars: set before importing torch/transformers
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
import copy
import random
import shutil
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

import optuna
from optuna.trial import TrialState
from optuna.storages import RDBStorage


# ============================================================
# Config
# ============================================================

@dataclass
class MLPHeadCfg:
    hidden_dim: int = 128
    layers: int = 1
    dropout: float = 0.1


@dataclass
class TrainCfg:
    seed: int = 42
    batch_size: int = 16
    max_len: int = 256

    # Safe reproducibility choice: use 0 workers everywhere
    num_workers: int = 0
    pin_memory: bool = True

    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1

    # Stage 1
    max_epochs_stage1: int = 500
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-3
    weight_decay_encoder: float = 0.01
    weight_decay_heads: float = 0.01
    finetune_top_k_layers: int = 1
    alpha_concept: float = 1.0
    warmup_ratio: float = 0.06

    # Stage 2
    max_epochs_stage2: int = 500
    lr_residual: float = 1e-3
    weight_decay_residual: float = 0.01
    lambda_resid_logits: float = 1e-3
    lambda_gate_mean: float = 1e-3
    warmup_ratio_stage2: float = 0.06

    patience: int = 10
    grad_accum_steps: int = 1

    save_explain: bool = True
    save_text_in_explain: bool = False
    topk_concepts: int = 20

    # log every N training steps to file
    log_every_n_steps: int = 50


@dataclass
class PathCfg:
    data_path: str = ""                     # set this
    local_bert_path: str = "./deberta"     # local DeBERTa
    output_dir: str = "./runs_cbm_residual_deberta_optuna"


@dataclass
class DataColsCfg:
    text_col: str = "text"
    target_col: str = "sentiment"


@dataclass
class FullCfg:
    paths: PathCfg
    cols: DataColsCfg
    train: TrainCfg
    concept_head: MLPHeadCfg
    residual_head: MLPHeadCfg
    gate_head: MLPHeadCfg


@dataclass
class ProgramCfg:
    total_trials: int = 50
    gpus: Tuple[int, int, int, int] = (0, 1, 6, 7)
    sqlite_timeout_sec: int = 60
    save_trial_artifacts: bool = False


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True)


def get_trial_seed(base_seed: int, trial_number: int) -> int:
    return int(base_seed + trial_number)


def make_loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def resolve_data_csv(data_path: str) -> str:
    p = Path(data_path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return str(p)
    if not p.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")
    if p.is_dir():
        csvs = sorted([x for x in p.glob("*.csv") if x.is_file()])
        if len(csvs) == 1:
            return str(csvs[0])
        raise ValueError(f"data_path='{data_path}' must contain exactly 1 CSV. Found {len(csvs)}")
    raise ValueError(f"data_path must be a CSV file or a directory. Got: {data_path}")


def make_logger(log_path: str) -> logging.Logger:
    ensure_dir(os.path.dirname(log_path) or ".")
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def freeze_all_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def detect_encoder_layers(model: nn.Module) -> Optional[List[nn.Module]]:
    candidates = [
        ("encoder.layer", lambda m: getattr(getattr(m, "encoder", None), "layer", None)),
        ("bert.encoder.layer", lambda m: getattr(getattr(getattr(m, "bert", None), "encoder", None), "layer", None)),
        ("roberta.encoder.layer", lambda m: getattr(getattr(getattr(m, "roberta", None), "encoder", None), "layer", None)),
        ("deberta.encoder.layer", lambda m: getattr(getattr(getattr(m, "deberta", None), "encoder", None), "layer", None)),
        ("deberta-v2.encoder.layer", lambda m: getattr(getattr(m, "encoder", None), "layer", None)),
        ("distilbert.transformer.layer", lambda m: getattr(getattr(getattr(m, "distilbert", None), "transformer", None), "layer", None)),
    ]
    for _, fn in candidates:
        layers = fn(model)
        if layers is not None:
            try:
                _ = len(layers)
                return list(layers)
            except Exception:
                pass
    return None


def unfreeze_top_k_transformer_layers(encoder: nn.Module, k: int) -> None:
    freeze_all_params(encoder)
    if k <= 0:
        return
    layers = detect_encoder_layers(encoder)
    if layers is None:
        raise RuntimeError("Cannot locate transformer layers. Adjust detect_encoder_layers().")
    k = min(k, len(layers))
    for layer in layers[-k:]:
        for p in layer.parameters():
            p.requires_grad = True
    for attr in ["pooler", "final_layer_norm", "layernorm", "LayerNorm"]:
        mod = getattr(encoder, attr, None)
        if mod is not None and isinstance(mod, nn.Module):
            for p in mod.parameters():
                p.requires_grad = True


def build_mlp(in_dim: int, out_dim: int, cfg: MLPHeadCfg) -> nn.Module:
    if cfg.layers < 0:
        raise ValueError("layers must be >= 0")
    if cfg.layers == 0:
        return nn.Linear(in_dim, out_dim)

    blocks: List[nn.Module] = []
    cur = in_dim
    for _ in range(cfg.layers):
        blocks.append(nn.Linear(cur, cfg.hidden_dim))
        blocks.append(nn.ReLU())
        if cfg.dropout > 0:
            blocks.append(nn.Dropout(cfg.dropout))
        cur = cfg.hidden_dim
    blocks.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*blocks)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    w = (len(y_train) / (num_classes * counts))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    rep = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        output_dict=True, zero_division=0
    )
    per_class = {}
    for c in range(num_classes):
        s = str(c)
        if s in rep:
            per_class[s] = {
                "precision": float(rep[s]["precision"]),
                "recall": float(rep[s]["recall"]),
                "f1": float(rep[s]["f1-score"]),
                "support": int(rep[s]["support"]),
            }
        else:
            per_class[s] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    out["per_class"] = per_class
    return out


def count_trainable_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def count_total_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def log_trainable_params_summary(logger: logging.Logger, model: nn.Module, title: str) -> None:
    total = count_total_params(model)
    trainable = count_trainable_params(model)
    pct = 100.0 * trainable / max(total, 1)
    logger.info(f"[{title}] trainable params = {trainable:,} / {total:,} ({pct:.2f}%)")
    if hasattr(model, "encoder"):
        enc_t = count_trainable_params(model.encoder)
        enc_all = count_total_params(model.encoder)
        logger.info(f"  - encoder      : {enc_t:,} / {enc_all:,} trainable")
    for name in ["concept_head", "label_head", "residual_head", "gate_head"]:
        if hasattr(model, name):
            m = getattr(model, name)
            logger.info(f"  - {name:<12}: {count_trainable_params(m):,} / {count_total_params(m):,} trainable")


# ============================================================
# Dataset
# ============================================================

class TextConceptDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        text_col: str,
        target_col: str,
        concept_cols: List[str],
        max_len: int,
        orig_idx_col: str = "_orig_idx",
        return_text: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.target_col = target_col
        self.concept_cols = concept_cols
        self.max_len = max_len
        self.orig_idx_col = orig_idx_col
        self.return_text = return_text

        if self.orig_idx_col not in self.df.columns:
            raise KeyError(f"Dataset missing required column: {self.orig_idx_col}")
        if len(concept_cols) == 0:
            raise ValueError("No concept columns found.")
        concept_mat = self.df[self.concept_cols]
        bad = [c for c in self.concept_cols if not np.issubdtype(concept_mat[c].dtype, np.number)]
        if bad:
            raise TypeError(f"All concept columns must be numeric. Bad cols: {bad[:10]}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row[self.text_col])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        y = int(row[self.target_col])
        c = row[self.concept_cols].to_numpy(dtype=np.float32)
        orig_idx = int(row[self.orig_idx_col])

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "concept_targets": torch.tensor(c, dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.long),
            "orig_idx": torch.tensor(orig_idx, dtype=torch.long),
        }
        if self.return_text:
            out["text"] = text
        return out


# ============================================================
# Model
# ============================================================

class CBMResidualModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_hidden_size: int,
        num_concepts: int,
        num_classes: int,
        concept_head_cfg: MLPHeadCfg,
        residual_head_cfg: MLPHeadCfg,
        gate_head_cfg: MLPHeadCfg,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.concept_head = build_mlp(encoder_hidden_size, num_concepts, concept_head_cfg)
        self.label_head = nn.Linear(num_concepts, num_classes, bias=True)
        self.residual_head = build_mlp(encoder_hidden_size, num_classes, residual_head_cfg)
        self.gate_head = build_mlp(encoder_hidden_size, 1, gate_head_cfg)
        self._phase = "cbm"

    def set_phase(self, phase: str) -> None:
        if phase not in ["cbm", "residual"]:
            raise ValueError("phase must be 'cbm' or 'residual'")
        self._phase = phase

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state[:, 0, :]
        elif isinstance(out, (tuple, list)):
            h = out[0][:, 0, :]
        else:
            raise RuntimeError("Unknown encoder output structure.")

        concepts_pred = self.concept_head(h)
        logits_cbm = self.label_head(concepts_pred)

        if self._phase == "cbm":
            resid_logits = torch.zeros_like(logits_cbm)
            gate = torch.zeros((logits_cbm.size(0), 1), device=logits_cbm.device, dtype=logits_cbm.dtype)
            logits = logits_cbm
        else:
            resid_logits = self.residual_head(h)
            gate = torch.sigmoid(self.gate_head(h))
            logits = logits_cbm + gate * resid_logits

        return {
            "concepts_pred": concepts_pred,
            "logits_cbm": logits_cbm,
            "resid_logits": resid_logits,
            "gate": gate,
            "logits": logits,
        }


# ============================================================
# Saving
# ============================================================

def save_run_globals(
    out_dir: str,
    cfg: FullCfg,
    label_encoder: LabelEncoder,
    concept_cols: List[str],
    split_indices: Dict[str, List[int]],
    concept_stats: Dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    save_json(os.path.join(out_dir, "config.json"), asdict(cfg))
    save_json(os.path.join(out_dir, "label_mapping.json"), {"classes_": label_encoder.classes_.tolist()})
    save_json(os.path.join(out_dir, "concept_columns.json"), {"concept_cols": concept_cols})
    save_json(os.path.join(out_dir, "split_indices.json"), split_indices)
    save_json(os.path.join(out_dir, "concept_stats.json"), concept_stats)


@torch.no_grad()
def dump_explainability(
    model: CBMResidualModel,
    loader: DataLoader,
    device: torch.device,
    phase: str,
    out_path: str,
    save_text: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    model.eval()
    model.set_phase(phase)

    orig_all, y_all, yhat_all = [], [], []
    logits_all, probs_all = [], []
    logits_cbm_all, resid_logits_all, gate_all = [], [], []
    c_pred_all, c_tgt_all = [], []
    text_all: List[str] = []

    num_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        orig_idx = batch["orig_idx"].to(device, non_blocking=True)
        c_tgt = batch["concept_targets"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1)
        yhat = torch.argmax(logits, dim=1)

        orig_all.append(orig_idx.detach().cpu())
        y_all.append(labels.detach().cpu())
        yhat_all.append(yhat.detach().cpu())
        logits_all.append(logits.detach().cpu())
        probs_all.append(probs.detach().cpu())
        logits_cbm_all.append(out["logits_cbm"].detach().cpu())
        resid_logits_all.append(out["resid_logits"].detach().cpu())
        gate_all.append(out["gate"].detach().cpu())
        c_pred_all.append(out["concepts_pred"].detach().cpu())
        c_tgt_all.append(c_tgt.detach().cpu())

        if save_text and ("text" in batch):
            text_all.extend(list(batch["text"]))

        if logger is not None and (step == num_steps or step % 50 == 0):
            logger.info(f"[dump:{phase}] step {step:04d}/{num_steps:04d}")

    payload: Dict[str, Any] = {
        "phase": phase,
        "orig_idx": torch.cat(orig_all, dim=0),
        "y_true": torch.cat(y_all, dim=0),
        "y_pred": torch.cat(yhat_all, dim=0),
        "logits": torch.cat(logits_all, dim=0),
        "probs": torch.cat(probs_all, dim=0),
        "logits_cbm": torch.cat(logits_cbm_all, dim=0),
        "resid_logits": torch.cat(resid_logits_all, dim=0),
        "gate": torch.cat(gate_all, dim=0),
        "concept_pred": torch.cat(c_pred_all, dim=0),
        "concept_target": torch.cat(c_tgt_all, dim=0),
    }
    if save_text:
        payload["text"] = text_all

    ensure_dir(os.path.dirname(out_path) or ".")
    torch.save(payload, out_path)


def export_label_head_weights(
    out_dir: str,
    model_module: CBMResidualModel,
    concept_cols: List[str],
    label_classes: List[str],
    topk: int = 20,
) -> None:
    ensure_dir(out_dir)

    W = model_module.label_head.weight.detach().cpu().numpy()
    b = model_module.label_head.bias.detach().cpu().numpy()

    np.save(os.path.join(out_dir, "label_head_weights.npy"), W)
    np.save(os.path.join(out_dir, "label_head_bias.npy"), b)

    dfW = pd.DataFrame(W, index=label_classes, columns=concept_cols)
    dfW.to_csv(os.path.join(out_dir, "label_head_weights.csv"), index=True)

    absW = np.abs(W)
    overall = absW.mean(axis=0)
    overall_idx = np.argsort(-overall)[:topk].tolist()

    topk_obj: Dict[str, Any] = {
        "topk": int(topk),
        "overall_mean_abs": [{"concept": concept_cols[i], "score": float(overall[i])} for i in overall_idx],
        "per_class_abs": {},
        "per_class_signed": {},
    }

    for ci, cname in enumerate(label_classes):
        idx_abs = np.argsort(-absW[ci])[:topk].tolist()
        idx_pos = np.argsort(-W[ci])[:topk].tolist()
        idx_neg = np.argsort(W[ci])[:topk].tolist()
        topk_obj["per_class_abs"][cname] = [
            {"concept": concept_cols[i], "score": float(absW[ci, i]), "w": float(W[ci, i])} for i in idx_abs
        ]
        topk_obj["per_class_signed"][cname] = {
            "top_positive": [{"concept": concept_cols[i], "w": float(W[ci, i])} for i in idx_pos],
            "top_negative": [{"concept": concept_cols[i], "w": float(W[ci, i])} for i in idx_neg],
        }

    save_json(os.path.join(out_dir, "label_head_topk.json"), topk_obj)


def save_stage_artifacts(
    out_dir: str,
    stage: str,
    model_state_dict: Dict[str, Any],
    metrics: Dict[str, Any],
    history: Dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    save_json(os.path.join(out_dir, f"best_metrics_{stage}.json"), metrics)
    save_json(os.path.join(out_dir, f"history_{stage}.json"), history)
    torch.save({"model_state_dict": model_state_dict}, os.path.join(out_dir, f"best_model_{stage}.pt"))


# ============================================================
# Train / Eval
# ============================================================

@torch.no_grad()
def run_eval(
    model: CBMResidualModel,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_weights: torch.Tensor,
    alpha_concept: float,
    phase: str,
) -> Dict[str, Any]:
    model.eval()
    model.set_phase(phase)

    ce = nn.CrossEntropyLoss(weight=class_weights)

    total_loss = 0.0
    total_concept_loss = 0.0
    total_label_loss = 0.0
    n = 0

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        concept_t = batch["concept_targets"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        concepts_pred = out["concepts_pred"]

        label_loss = ce(logits, labels)
        concept_loss = F.mse_loss(concepts_pred, concept_t)
        loss = label_loss + alpha_concept * concept_loss

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_concept_loss += float(concept_loss.item()) * bs
        total_label_loss += float(label_loss.item()) * bs
        n += bs

        preds = torch.argmax(logits, dim=1)
        y_true_all.extend(labels.detach().cpu().tolist())
        y_pred_all.extend(preds.detach().cpu().tolist())

    n = max(int(n), 1)
    metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all), num_classes=num_classes)
    metrics["loss"] = float(total_loss / n)
    metrics["concept_loss"] = float(total_concept_loss / n)
    metrics["label_loss"] = float(total_label_loss / n)
    return metrics


def run_train_one_epoch(
    model: CBMResidualModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    num_classes: int,
    class_weights: torch.Tensor,
    alpha_concept: float,
    phase: str,
    lambda_resid_logits: float,
    lambda_gate_mean: float,
    grad_accum_steps: int,
    logger: logging.Logger,
    stage_name: str,
    epoch: int,
    log_every_n_steps: int,
) -> Dict[str, Any]:
    model.train()
    model.set_phase(phase)

    ce = nn.CrossEntropyLoss(weight=class_weights)

    total_loss = 0.0
    total_concept_loss = 0.0
    total_label_loss = 0.0
    total_resid_pen = 0.0
    total_gate_pen = 0.0
    n = 0

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    optimizer.zero_grad(set_to_none=True)

    num_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        concept_t = batch["concept_targets"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        concepts_pred = out["concepts_pred"]

        label_loss = ce(logits, labels)
        concept_loss = F.mse_loss(concepts_pred, concept_t)

        resid_logits = out["resid_logits"]
        gate = out["gate"]

        if phase == "residual":
            resid_pen = (resid_logits ** 2).mean()
            gate_pen = gate.mean()
        else:
            resid_pen = torch.tensor(0.0, device=device)
            gate_pen = torch.tensor(0.0, device=device)

        loss = label_loss + alpha_concept * concept_loss + lambda_resid_logits * resid_pen + lambda_gate_mean * gate_pen
        (loss / grad_accum_steps).backward()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_label_loss += float(label_loss.item()) * bs
        total_concept_loss += float(concept_loss.item()) * bs
        total_resid_pen += float(resid_pen.item()) * bs
        total_gate_pen += float(gate_pen.item()) * bs
        n += bs

        preds = torch.argmax(logits, dim=1)
        y_true_all.extend(labels.detach().cpu().tolist())
        y_pred_all.extend(preds.detach().cpu().tolist())

        sync_now = (step % grad_accum_steps == 0) or (step == num_steps)
        if sync_now:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        if step == num_steps or step % log_every_n_steps == 0:
            live_acc = accuracy_score(y_true_all, y_pred_all) if n > 0 else 0.0
            live_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0) if n > 0 else 0.0
            logger.info(
                f"[{stage_name}] epoch {epoch:03d} step {step:04d}/{num_steps:04d} | "
                f"loss={total_loss / max(n,1):.4f} acc={live_acc:.4f} macro_f1={live_f1:.4f}"
            )

    n = max(int(n), 1)
    metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all), num_classes=num_classes)
    metrics["loss"] = float(total_loss / n)
    metrics["label_loss"] = float(total_label_loss / n)
    metrics["concept_loss"] = float(total_concept_loss / n)
    metrics["resid_pen"] = float(total_resid_pen / n)
    metrics["gate_pen"] = float(total_gate_pen / n)
    return metrics


def make_optimizer_and_scheduler(
    model_module: CBMResidualModel,
    phase: str,
    train_cfg: TrainCfg,
    steps_per_epoch: int,
    total_epochs: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    def params_with_decay(named_params, weight_decay: float):
        decay, no_decay = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias") or "LayerNorm" in n or "layer_norm" in n or "ln" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    if phase == "cbm":
        enc_groups = params_with_decay(model_module.encoder.named_parameters(), train_cfg.weight_decay_encoder)
        head_named = list(model_module.concept_head.named_parameters()) + list(model_module.label_head.named_parameters())
        head_groups = params_with_decay(head_named, train_cfg.weight_decay_heads)
        for g in enc_groups:
            g["lr"] = train_cfg.lr_encoder
        for g in head_groups:
            g["lr"] = train_cfg.lr_heads
        param_groups = enc_groups + head_groups
        warmup_ratio = train_cfg.warmup_ratio
    else:
        resid_groups = params_with_decay(model_module.residual_head.named_parameters(), train_cfg.weight_decay_residual)
        gate_groups = params_with_decay(model_module.gate_head.named_parameters(), train_cfg.weight_decay_residual)
        for g in resid_groups:
            g["lr"] = train_cfg.lr_residual
        for g in gate_groups:
            g["lr"] = train_cfg.lr_residual * 2.0
        param_groups = resid_groups + gate_groups
        warmup_ratio = train_cfg.warmup_ratio_stage2

    optimizer = torch.optim.AdamW(param_groups)
    total_steps = max(1, steps_per_epoch * total_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def early_stopping_step(
    best_score: float,
    current_score: float,
    no_improve: int,
    patience: int,
) -> Tuple[bool, float, int]:
    if current_score > best_score + 1e-12:
        return False, current_score, 0
    no_improve += 1
    return (no_improve >= patience), best_score, no_improve


def plot_losses_stage2(
    out_dir: str,
    stage: str,
    train_overall: List[float],
    val_overall: List[float],
    train_concept: List[float],
    val_concept: List[float],
) -> None:
    ensure_dir(out_dir)
    plt.figure()
    plt.plot(range(1, len(train_overall) + 1), train_overall, label="train")
    plt.plot(range(1, len(val_overall) + 1), val_overall, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Overall loss")
    plt.title(f"{stage}: overall loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stage}_loss_overall.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_concept) + 1), train_concept, label="train")
    plt.plot(range(1, len(val_concept) + 1), val_concept, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Concept loss (MSE)")
    plt.title(f"{stage}: concept loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stage}_loss_concept.png"))
    plt.close()


def plot_stage1_combined(
    out_dir: str,
    stage: str,
    train_overall: List[float],
    val_overall: List[float],
    train_concept: List[float],
    val_concept: List[float],
    train_label: List[float],
    val_label: List[float],
) -> None:
    ensure_dir(out_dir)
    epochs = range(1, len(train_overall) + 1)
    plt.figure()
    plt.plot(epochs, train_overall, label="train_overall")
    plt.plot(epochs, val_overall, label="val_overall")
    plt.plot(epochs, train_concept, label="train_concept")
    plt.plot(epochs, val_concept, label="val_concept")
    plt.plot(epochs, train_label, label="train_label")
    plt.plot(epochs, val_label, label="val_label")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{stage}: overall vs concept vs label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stage}_loss_all_in_one.png"))
    plt.close()


def build_splits(
    df: pd.DataFrame,
    target_col: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not math.isclose(train_size + val_size + test_size, 1.0, rel_tol=1e-6):
        raise ValueError("train/val/test sizes must sum to 1")
    df_train, df_tmp = train_test_split(df, train_size=train_size, random_state=seed, stratify=df[target_col])
    tmp_ratio = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(df_tmp, train_size=tmp_ratio, random_state=seed, stratify=df_tmp[target_col])
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def train_stage(
    stage_name: str,
    model: CBMResidualModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: FullCfg,
    num_classes: int,
    class_weights: torch.Tensor,
    phase: str,
    max_epochs: int,
    out_dir: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.train.grad_accum_steps))
    optimizer, scheduler = make_optimizer_and_scheduler(
        model_module=model,
        phase=phase,
        train_cfg=cfg.train,
        steps_per_epoch=steps_per_epoch,
        total_epochs=max_epochs,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0

    train_overall_hist: List[float] = []
    val_overall_hist: List[float] = []
    train_concept_hist: List[float] = []
    val_concept_hist: List[float] = []
    train_label_hist: List[float] = []
    val_label_hist: List[float] = []

    best_state: Optional[Dict[str, Any]] = None
    best_train_metrics: Optional[Dict[str, Any]] = None
    best_val_metrics: Optional[Dict[str, Any]] = None

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=num_classes,
            class_weights=class_weights,
            alpha_concept=cfg.train.alpha_concept,
            phase=phase,
            lambda_resid_logits=cfg.train.lambda_resid_logits,
            lambda_gate_mean=cfg.train.lambda_gate_mean,
            grad_accum_steps=cfg.train.grad_accum_steps,
            logger=logger,
            stage_name=stage_name,
            epoch=epoch,
            log_every_n_steps=cfg.train.log_every_n_steps,
        )
        val_metrics = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            class_weights=class_weights,
            alpha_concept=cfg.train.alpha_concept,
            phase=phase,
        )

        train_overall_hist.append(train_metrics["loss"])
        val_overall_hist.append(val_metrics["loss"])
        train_concept_hist.append(train_metrics["concept_loss"])
        val_concept_hist.append(val_metrics["concept_loss"])
        train_label_hist.append(train_metrics["label_loss"])
        val_label_hist.append(val_metrics["label_loss"])

        logger.info(
            f"[{stage_name}] epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"train_label={train_metrics['label_loss']:.4f} val_label={val_metrics['label_loss']:.4f} | "
            f"train_concept={train_metrics['concept_loss']:.4f} val_concept={val_metrics['concept_loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        stop, best_val_f1, no_improve = early_stopping_step(
            best_score=best_val_f1,
            current_score=float(val_metrics["macro_f1"]),
            no_improve=no_improve,
            patience=cfg.train.patience,
        )

        if float(val_metrics["macro_f1"]) >= best_val_f1 - 1e-12 and no_improve == 0:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_train_metrics = copy.deepcopy(train_metrics)
            best_val_metrics = copy.deepcopy(val_metrics)

        if stop:
            logger.info(
                f"[{stage_name}] early stop at epoch {epoch} "
                f"(best epoch={best_epoch}, best val macro_f1={best_val_f1:.4f})"
            )
            break

    if best_state is None:
        raise RuntimeError(f"{stage_name}: best_state is None")
    model.load_state_dict(best_state)

    bundle = {
        "best_epoch": best_epoch,
        "best_state_dict": best_state,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "history": {
            "train_overall": train_overall_hist,
            "val_overall": val_overall_hist,
            "train_concept": train_concept_hist,
            "val_concept": val_concept_hist,
            "train_label": train_label_hist,
            "val_label": val_label_hist,
        },
    }

    if stage_name == "stage1_cbm":
        plot_stage1_combined(
            out_dir=out_dir,
            stage=stage_name,
            train_overall=train_overall_hist,
            val_overall=val_overall_hist,
            train_concept=train_concept_hist,
            val_concept=val_concept_hist,
            train_label=train_label_hist,
            val_label=val_label_hist,
        )
    else:
        plot_losses_stage2(
            out_dir=out_dir,
            stage=stage_name,
            train_overall=train_overall_hist,
            val_overall=val_overall_hist,
            train_concept=train_concept_hist,
            val_concept=val_concept_hist,
        )

    return bundle


# ============================================================
# End-to-end experiment
# ============================================================

def run_single_experiment(cfg: FullCfg, logger: logging.Logger) -> Dict[str, Any]:
    if not cfg.paths.data_path:
        raise ValueError("cfg.paths.data_path is empty. Set it to a CSV file or a directory containing one CSV.")

    set_seed(cfg.train.seed)

    data_csv = resolve_data_csv(cfg.paths.data_path)
    ensure_dir(cfg.paths.output_dir)

    df = pd.read_csv(data_csv)

    if cfg.cols.text_col not in df.columns:
        raise KeyError(f"text_col '{cfg.cols.text_col}' not found in df.columns")
    if cfg.cols.target_col not in df.columns:
        raise KeyError(f"target_col '{cfg.cols.target_col}' not found in df.columns")

    concept_cols = [c for c in df.columns if c not in [cfg.cols.text_col, cfg.cols.target_col]]
    if len(concept_cols) == 0:
        raise ValueError("No concept columns found (expected columns besides text/target).")

    bad = [c for c in concept_cols if not np.issubdtype(df[c].dtype, np.number)]
    if bad:
        raise TypeError(f"Non-numeric concept columns detected: {bad[:10]} (convert before training).")

    le = LabelEncoder()
    df[cfg.cols.target_col] = le.fit_transform(df[cfg.cols.target_col].astype(str))
    num_classes = int(df[cfg.cols.target_col].nunique())
    if num_classes < 2:
        raise ValueError(f"Need >=2 classes; got {num_classes}")

    df = df.reset_index(drop=True)
    df["_orig_idx"] = np.arange(len(df), dtype=np.int64)

    df_train, df_val, df_test = build_splits(
        df=df,
        target_col=cfg.cols.target_col,
        train_size=cfg.train.train_size,
        val_size=cfg.train.val_size,
        test_size=cfg.train.test_size,
        seed=cfg.train.seed,
    )

    split_indices = {
        "train": df_train["_orig_idx"].tolist(),
        "val": df_val["_orig_idx"].tolist(),
        "test": df_test["_orig_idx"].tolist(),
    }

    concept_stats = {
        "mean": df_train[concept_cols].mean().to_dict(),
        "std": df_train[concept_cols].std(ddof=0).replace(0, 1.0).to_dict(),
        "min": df_train[concept_cols].min().to_dict(),
        "max": df_train[concept_cols].max().to_dict(),
    }

    save_run_globals(
        out_dir=cfg.paths.output_dir,
        cfg=cfg,
        label_encoder=le,
        concept_cols=concept_cols,
        split_indices=split_indices,
        concept_stats=concept_stats,
    )

    y_train_np = df_train[cfg.cols.target_col].to_numpy(dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_w = compute_class_weights(y_train_np, num_classes=num_classes).to(device)

    logger.info(f"Using device={device}")
    logger.info(f"data_csv={data_csv}")
    logger.info(f"num_classes={num_classes} num_concepts={len(concept_cols)}")
    logger.info(f"seed={cfg.train.seed}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.paths.local_bert_path,
        local_files_only=True,
        use_fast=False,
    )
    encoder = AutoModel.from_pretrained(
        cfg.paths.local_bert_path,
        local_files_only=True,
    )

    hs = getattr(encoder.config, "hidden_size", None)
    if hs is None:
        hs = getattr(encoder.config, "dim", None)
    if hs is None:
        raise RuntimeError("Cannot infer encoder hidden size from encoder.config.hidden_size or .dim")
    hidden_size = int(hs)

    unfreeze_top_k_transformer_layers(encoder, cfg.train.finetune_top_k_layers)

    train_ds = TextConceptDataset(
        df_train, tokenizer, cfg.cols.text_col, cfg.cols.target_col, concept_cols, cfg.train.max_len,
        orig_idx_col="_orig_idx", return_text=False
    )
    val_ds = TextConceptDataset(
        df_val, tokenizer, cfg.cols.text_col, cfg.cols.target_col, concept_cols, cfg.train.max_len,
        orig_idx_col="_orig_idx", return_text=False
    )
    test_ds = TextConceptDataset(
        df_test, tokenizer, cfg.cols.text_col, cfg.cols.target_col, concept_cols, cfg.train.max_len,
        orig_idx_col="_orig_idx", return_text=bool(cfg.train.save_text_in_explain)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        generator=make_loader_generator(cfg.train.seed + 101),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        generator=make_loader_generator(cfg.train.seed + 102),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        generator=make_loader_generator(cfg.train.seed + 103),
    )

    model = CBMResidualModel(
        encoder=encoder,
        encoder_hidden_size=hidden_size,
        num_concepts=len(concept_cols),
        num_classes=num_classes,
        concept_head_cfg=cfg.concept_head,
        residual_head_cfg=cfg.residual_head,
        gate_head_cfg=cfg.gate_head,
    ).to(device)

    # Stage 1
    stage1_dir = os.path.join(cfg.paths.output_dir, "stage1_cbm")
    ensure_dir(stage1_dir)

    freeze_all_params(model.residual_head)
    freeze_all_params(model.gate_head)
    log_trainable_params_summary(logger, model, "BEGIN stage1_cbm")

    bundle1 = train_stage(
        stage_name="stage1_cbm",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        num_classes=num_classes,
        class_weights=class_w,
        phase="cbm",
        max_epochs=cfg.train.max_epochs_stage1,
        out_dir=stage1_dir,
        logger=logger,
    )

    train_m1 = run_eval(model, train_loader, device, num_classes, class_w, cfg.train.alpha_concept, phase="cbm")
    val_m1 = run_eval(model, val_loader, device, num_classes, class_w, cfg.train.alpha_concept, phase="cbm")
    test_m1 = run_eval(model, test_loader, device, num_classes, class_w, cfg.train.alpha_concept, phase="cbm")

    metrics_stage1 = {"best_epoch": bundle1["best_epoch"], "train": train_m1, "val": val_m1, "test": test_m1}
    save_stage_artifacts(
        out_dir=stage1_dir,
        stage="stage1_cbm",
        model_state_dict=model.state_dict(),
        metrics=metrics_stage1,
        history=bundle1["history"],
    )

    export_label_head_weights(
        out_dir=os.path.join(cfg.paths.output_dir, "analysis"),
        model_module=model,
        concept_cols=concept_cols,
        label_classes=le.classes_.tolist(),
        topk=int(cfg.train.topk_concepts),
    )

    if cfg.train.save_explain:
        dump_explainability(
            model=model,
            loader=test_loader,
            device=device,
            phase="cbm",
            out_path=os.path.join(cfg.paths.output_dir, "explainability", "test_dump_cbm.pt"),
            save_text=bool(cfg.train.save_text_in_explain),
            logger=logger,
        )

    # Stage 2
    stage2_dir = os.path.join(cfg.paths.output_dir, "stage2_residual")
    ensure_dir(stage2_dir)

    freeze_all_params(model.encoder)
    freeze_all_params(model.concept_head)
    freeze_all_params(model.label_head)
    for p in model.residual_head.parameters():
        p.requires_grad = True
    for p in model.gate_head.parameters():
        p.requires_grad = True

    log_trainable_params_summary(logger, model, "BEGIN stage2_residual")

    bundle2 = train_stage(
        stage_name="stage2_residual",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        num_classes=num_classes,
        class_weights=class_w,
        phase="residual",
        max_epochs=cfg.train.max_epochs_stage2,
        out_dir=stage2_dir,
        logger=logger,
    )

    train_m2 = run_eval(model, train_loader, device, num_classes, class_w, cfg.train.alpha_concept, phase="residual")
    val_m2 = run_eval(model, val_loader, device, num_classes, class_w, cfg.train.alpha_concept, phase="residual")
    test_m2 = run_eval(model, test_loader, device, num_classes, class_w, cfg.train.alpha_concept, phase="residual")

    metrics_stage2 = {
        "best_epoch": bundle2["best_epoch"],
        "train": train_m2,
        "val": val_m2,
        "test": test_m2,
        "residual_regularization": {
            "lambda_resid_logits": cfg.train.lambda_resid_logits,
            "lambda_gate_mean": cfg.train.lambda_gate_mean,
        },
    }
    save_stage_artifacts(
        out_dir=stage2_dir,
        stage="stage2_residual",
        model_state_dict=model.state_dict(),
        metrics=metrics_stage2,
        history=bundle2["history"],
    )

    save_json(
        os.path.join(cfg.paths.output_dir, "metrics_summary.json"),
        {
            "stage": "stage2_residual",
            "best_epoch": int(bundle2["best_epoch"]),
            "train_macro_f1": float(train_m2["macro_f1"]),
            "val_macro_f1": float(val_m2["macro_f1"]),
            "test_macro_f1": float(test_m2["macro_f1"]),
        },
    )

    if cfg.train.save_explain:
        dump_explainability(
            model=model,
            loader=test_loader,
            device=device,
            phase="residual",
            out_path=os.path.join(cfg.paths.output_dir, "explainability", "test_dump_residual.pt"),
            save_text=bool(cfg.train.save_text_in_explain),
            logger=logger,
        )

    save_json(
        os.path.join(cfg.paths.output_dir, "FINAL_POINTER.json"),
        {
            "final_stage": "stage2_residual",
            "final_model_path": os.path.join(stage2_dir, "best_model_stage2_residual.pt"),
            "final_metrics_path": os.path.join(stage2_dir, "best_metrics_stage2_residual.json"),
            "explain_dump_cbm_path": (
                os.path.join(cfg.paths.output_dir, "explainability", "test_dump_cbm.pt")
                if cfg.train.save_explain else None
            ),
            "explain_dump_residual_path": (
                os.path.join(cfg.paths.output_dir, "explainability", "test_dump_residual.pt")
                if cfg.train.save_explain else None
            ),
        },
    )

    logger.info("Done.")
    logger.info(f"Run root: {cfg.paths.output_dir}")
    logger.info(f"Stage1:   {stage1_dir}")
    logger.info(f"Stage2:   {stage2_dir}")

    return {
        "val_macro_f1": float(val_m2["macro_f1"]),
        "test_macro_f1": float(test_m2["macro_f1"]),
        "train_macro_f1": float(train_m2["macro_f1"]),
        "final_stage": "stage2_residual",
        "run_root": cfg.paths.output_dir,
    }


# ============================================================
# Optuna
# ============================================================

def suggest_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    hp: Dict[str, Any] = {}
    hp["finetune_top_k_layers"] = trial.suggest_categorical("finetune_top_k_layers", [0, 1, 2, 4, 6, 8, 12])

    hp["lr_encoder"] = trial.suggest_categorical("lr_encoder", [1e-5, 2e-5, 3e-5, 5e-5])
    hp["lr_heads"] = trial.suggest_categorical("lr_heads", [1e-4, 2e-4, 5e-4, 1e-3])
    hp["lr_residual"] = trial.suggest_categorical("lr_residual", [1e-4, 2e-4, 5e-4, 1e-3])

    hp["weight_decay_encoder"] = trial.suggest_categorical("weight_decay_encoder", [0.0, 0.01, 0.02, 0.05])
    hp["weight_decay_heads"] = trial.suggest_categorical("weight_decay_heads", [0.0, 0.01, 0.02, 0.05])
    hp["weight_decay_residual"] = trial.suggest_categorical("weight_decay_residual", [0.0, 0.01, 0.02, 0.05])

    hp["warmup_ratio"] = trial.suggest_categorical("warmup_ratio", [0.0, 0.03, 0.06, 0.10, 0.12])
    hp["warmup_ratio_stage2"] = trial.suggest_categorical("warmup_ratio_stage2", [0.0, 0.03, 0.06, 0.10, 0.12])

    hp["concept_hidden_dim"] = trial.suggest_categorical("concept_hidden_dim", [64, 128, 256])
    hp["concept_layers"] = trial.suggest_categorical("concept_layers", [1, 2, 3])
    hp["concept_dropout"] = trial.suggest_categorical("concept_dropout", [0.0, 0.1, 0.2, 0.3])

    hp["residual_hidden_dim"] = trial.suggest_categorical("residual_hidden_dim", [64, 128, 256])
    hp["residual_layers"] = trial.suggest_categorical("residual_layers", [1, 2, 3])
    hp["residual_dropout"] = trial.suggest_categorical("residual_dropout", [0.0, 0.1, 0.2, 0.3])

    hp["gate_hidden_dim"] = trial.suggest_categorical("gate_hidden_dim", [32, 64, 128, 256])
    hp["gate_layers"] = trial.suggest_categorical("gate_layers", [1, 2, 3])
    hp["gate_dropout"] = trial.suggest_categorical("gate_dropout", [0.0, 0.1, 0.2, 0.3])

    hp["alpha_concept"] = trial.suggest_float("alpha_concept", 1.0, 15.0)
    return hp


def apply_hparams(cfg: FullCfg, hp: Dict[str, Any]) -> FullCfg:
    cfg = copy.deepcopy(cfg)
    cfg.train.finetune_top_k_layers = int(hp["finetune_top_k_layers"])
    cfg.train.lr_encoder = float(hp["lr_encoder"])
    cfg.train.lr_heads = float(hp["lr_heads"])
    cfg.train.lr_residual = float(hp["lr_residual"])
    cfg.train.weight_decay_encoder = float(hp["weight_decay_encoder"])
    cfg.train.weight_decay_heads = float(hp["weight_decay_heads"])
    cfg.train.weight_decay_residual = float(hp["weight_decay_residual"])
    cfg.train.warmup_ratio = float(hp["warmup_ratio"])
    cfg.train.warmup_ratio_stage2 = float(hp["warmup_ratio_stage2"])
    cfg.train.alpha_concept = float(hp["alpha_concept"])

    cfg.concept_head.hidden_dim = int(hp["concept_hidden_dim"])
    cfg.concept_head.layers = int(hp["concept_layers"])
    cfg.concept_head.dropout = float(hp["concept_dropout"])

    cfg.residual_head.hidden_dim = int(hp["residual_hidden_dim"])
    cfg.residual_head.layers = int(hp["residual_layers"])
    cfg.residual_head.dropout = float(hp["residual_dropout"])

    cfg.gate_head.hidden_dim = int(hp["gate_hidden_dim"])
    cfg.gate_head.layers = int(hp["gate_layers"])
    cfg.gate_head.dropout = float(hp["gate_dropout"])
    return cfg


def worker_optimize(
    full_cfg: FullCfg,
    program_cfg: ProgramCfg,
    worker_id: int,
    gpu_id: int,
) -> None:
    torch.cuda.set_device(int(gpu_id))
    logger = make_logger(os.path.join(full_cfg.paths.output_dir, f"worker{worker_id}_gpu{gpu_id}.log"))
    logger.info(f"[WORKER {worker_id}] start | device=cuda:{gpu_id}")

    storage_url = f"sqlite:///{os.path.join(full_cfg.paths.output_dir, 'study.sqlite3')}"
    study = optuna.load_study(study_name="cbm_residual_single_gpu_optuna", storage=storage_url)

    max_cb = optuna.study.MaxTrialsCallback(
        n_trials=program_cfg.total_trials,
        states=(TrialState.COMPLETE,),
    )

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hparams(trial)

        trial_dir = os.path.join(full_cfg.paths.output_dir, "_tmp_optuna_trials", f"trial_{trial.number:03d}")
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)
        ensure_dir(trial_dir)

        cfg_trial = apply_hparams(full_cfg, hp)
        cfg_trial.paths.output_dir = trial_dir
        cfg_trial.train.save_explain = False
        cfg_trial.train.save_text_in_explain = False

        trial_seed = get_trial_seed(full_cfg.train.seed, trial.number)
        cfg_trial.train.seed = trial_seed
        set_seed(trial_seed)

        logger.info(f"[TRIAL {trial.number}] start | gpu={gpu_id} | trial_seed={trial_seed} | hp={hp}")
        result = run_single_experiment(cfg_trial, logger)
        val_f1 = float(result["val_macro_f1"])
        logger.info(f"[TRIAL {trial.number}] end | val_macro_f1={val_f1:.6f}")

        if not program_cfg.save_trial_artifacts:
            shutil.rmtree(trial_dir, ignore_errors=True)

        return val_f1

    study.optimize(objective, n_trials=10_000, callbacks=[max_cb], gc_after_trial=True)
    logger.info(f"[WORKER {worker_id}] done")


def init_storage_and_study(output_dir: str, sqlite_timeout_sec: int) -> str:
    study_path = os.path.join(output_dir, "study.sqlite3")
    if os.path.exists(study_path):
        os.remove(study_path)
    storage_url = f"sqlite:///{study_path}"

    storage = RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": int(sqlite_timeout_sec)}},
    )

    optuna.create_study(
        study_name="cbm_residual_single_gpu_optuna",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )
    return storage_url


def run_parallel_optuna(full_cfg: FullCfg, program_cfg: ProgramCfg) -> Dict[str, Any]:
    import multiprocessing as mp

    ensure_dir(full_cfg.paths.output_dir)
    tmp_trials_dir = os.path.join(full_cfg.paths.output_dir, "_tmp_optuna_trials")
    ensure_dir(tmp_trials_dir)

    storage_url = init_storage_and_study(full_cfg.paths.output_dir, program_cfg.sqlite_timeout_sec)

    ctx = mp.get_context("spawn")
    procs = []
    for wid, gid in enumerate(program_cfg.gpus):
        p = ctx.Process(target=worker_optimize, args=(full_cfg, program_cfg, wid, gid), daemon=False)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    study = optuna.load_study(study_name="cbm_residual_single_gpu_optuna", storage=storage_url)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) < program_cfg.total_trials:
        raise RuntimeError(f"Study ended with fewer completed trials than expected: {len(completed)} < {program_cfg.total_trials}")

    best = study.best_trial
    best_payload = {
        "best_value": float(best.value),
        "best_trial_number": int(best.number),
        "best_trial_seed": int(get_trial_seed(full_cfg.train.seed, best.number)),
        "best_params": dict(best.params),
    }
    save_json(os.path.join(full_cfg.paths.output_dir, "best_optuna_params.json"), best_payload)

    if not program_cfg.save_trial_artifacts:
        shutil.rmtree(tmp_trials_dir, ignore_errors=True)

    return best_payload


def train_and_save_best(
    full_cfg: FullCfg,
    best_params: Dict[str, Any],
    best_trial_seed: int,
    program_cfg: ProgramCfg,
) -> str:
    best_dir = full_cfg.paths.output_dir

    for name in [
        "analysis",
        "explainability",
        "stage1_cbm",
        "stage2_residual",
    ]:
        path = os.path.join(best_dir, name)
        if os.path.exists(path):
            shutil.rmtree(path)

    for name in [
        "best_model.pt",
        "config.json",
        "label_mapping.json",
        "concept_columns.json",
        "split_indices.json",
        "concept_stats.json",
        "metrics_summary.json",
        "FINAL_POINTER.json",
        "final_best_summary.json",
    ]:
        path = os.path.join(best_dir, name)
        if os.path.exists(path):
            os.remove(path)

    cfg_best = apply_hparams(full_cfg, best_params)
    cfg_best.paths.output_dir = best_dir
    cfg_best.train.save_explain = True
    cfg_best.train.seed = int(best_trial_seed)

    set_seed(cfg_best.train.seed)

    gpu_id = int(program_cfg.gpus[0])
    torch.cuda.set_device(gpu_id)
    logger = make_logger(os.path.join(full_cfg.paths.output_dir, f"best_run_gpu{gpu_id}.log"))
    logger.info(f"[BEST] start | gpu={gpu_id} | best_trial_seed={best_trial_seed} | params={best_params}")

    _ = run_single_experiment(cfg_best, logger)

    final_stage_model = os.path.join(best_dir, "stage2_residual", "best_model_stage2_residual.pt")
    if os.path.exists(final_stage_model):
        shutil.copy2(final_stage_model, os.path.join(best_dir, "best_model.pt"))

    stage2_metrics_path = os.path.join(best_dir, "stage2_residual", "best_metrics_stage2_residual.json")
    if not os.path.exists(stage2_metrics_path):
        raise FileNotFoundError(f"Missing expected metrics file: {stage2_metrics_path}")

    with open(stage2_metrics_path, "r", encoding="utf-8") as f:
        stage2_metrics = json.load(f)

    save_json(
        os.path.join(full_cfg.paths.output_dir, "final_best_summary.json"),
        {
            "best_epoch": int(stage2_metrics["best_epoch"]),
            "best_train_macro_f1": float(stage2_metrics["train"]["macro_f1"]),
            "best_val_macro_f1": float(stage2_metrics["val"]["macro_f1"]),
            "best_test_macro_f1": float(stage2_metrics["test"]["macro_f1"]),
            "best_train_accuracy": float(stage2_metrics["train"]["accuracy"]),
            "best_val_accuracy": float(stage2_metrics["val"]["accuracy"]),
            "best_test_accuracy": float(stage2_metrics["test"]["accuracy"]),
            "best_params": best_params,
            "best_trial_seed": int(best_trial_seed),
            "source_of_truth": stage2_metrics_path,
            "run_dir": best_dir,
        },
    )

    logger.info(f"[BEST] done | run_dir={best_dir}")
    return best_dir


# ============================================================
# Program defaults (no CLI)
# ============================================================

def build_default_full_cfg() -> FullCfg:
    return FullCfg(
        paths=PathCfg(
            data_path="/data/users/zhangtianxiao/suicide/data/suicide500_evidence_aware_concepts72.csv",  # set this to your CSV file or a directory containing exactly one CSV
            local_bert_path="/data/users/zhangtianxiao/mental_llm_mas/models/deberta",
            output_dir="./runs_cbm_residual_deberta_optuna_suicide500",
        ),
        cols=DataColsCfg(text_col="text", target_col="label"),
        train=TrainCfg(),
        concept_head=MLPHeadCfg(),
        residual_head=MLPHeadCfg(),
        gate_head=MLPHeadCfg(),
    )


def build_default_program_cfg() -> ProgramCfg:
    return ProgramCfg(
        total_trials=50,
        gpus=(0, 1, 4, 5, 6, 7),
        sqlite_timeout_sec=60,
        save_trial_artifacts=False,
    )


def main() -> None:
    full_cfg = build_default_full_cfg()
    program_cfg = build_default_program_cfg()

    if not full_cfg.paths.data_path:
        raise ValueError(
            "Please set build_default_full_cfg().paths.data_path to your CSV file "
            "or a directory containing exactly one CSV."
        )

    ensure_dir(full_cfg.paths.output_dir)

    best = run_parallel_optuna(full_cfg, program_cfg)
    run_dir = train_and_save_best(
        full_cfg=full_cfg,
        best_params=best["best_params"],
        best_trial_seed=int(best["best_trial_seed"]),
        program_cfg=program_cfg,
    )

    print(f"[DONE] Best val macro-F1={best['best_value']:.6f}")
    print(f"[DONE] Best artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()