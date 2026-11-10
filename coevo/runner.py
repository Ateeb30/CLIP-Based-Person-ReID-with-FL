"""End-to-end run: Phase I anchors, style bank, coupled federated loop, evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .config import Config, dump_config
from .data import (build_eval_loader, build_test_transform, build_train_transform, load_dataset,
                   missing_files)
from .evaluation import evaluate_dataset
from .fed import Client, build_aggregator
from .models import build_clip, build_image_encoder
from .phase1 import learn_anchors
from .style import GlobalStyleDiversification, StyleBank, extract_camera_templates
from .utils import (MetricWriter, atomic_save, git_info, resolve_device, rng_state, seed_everything,
                    set_rng_state, setup_logger)

REPO_DIR = Path(__file__).resolve().parent.parent


def round_lr(fed_cfg, round_index: int) -> float:
    """MultiStepLR over communication rounds; `round_index` is 0-based."""
    passed = sum(1 for m in fed_cfg.lr_milestones if round_index >= m)
    return fed_cfg.optimizer.lr * fed_cfg.lr_gamma ** passed


def load_all_data(cfg, logger) -> tuple[dict, dict]:
    sources = {name: load_dataset(name, cfg.data) for name in cfg.data.sources}
    targets = {name: sources[name] if name in sources else load_dataset(name, cfg.data)
               for name in cfg.data.targets}
    problems = []
    for name, data in sources.items():
        logger.info(data.summary())
        missing = missing_files(data.train)
        if missing:
            problems.append(f"{name} train: {len(missing)} of {len(data.train)} images missing, e.g. {missing[0]}")
    for name, data in targets.items():
        if name not in sources:
            logger.info(data.summary())
        missing = missing_files(data.query + data.gallery)
        if missing:
            problems.append(f"{name} test: {len(missing)} of {len(data.query) + len(data.gallery)} "
                            f"images missing, e.g. {missing[0]}")
    if problems:
        raise FileNotFoundError("incomplete datasets:\n  " + "\n  ".join(problems))
    return sources, targets


def build_style_bank(cfg, sources, transform, device, logger) -> StyleBank:
    templates = {}
    for name, data in sources.items():
        start = time.time()
        loader = build_eval_loader(data.train, transform, cfg.style.bank_batch_size, cfg.data.num_workers,
                                   pin_memory=device.type == "cuda")
        templates[name] = extract_camera_templates(loader, device)
        logger.info(f"[bank] {name}: {len(templates[name])} camera templates in {time.time() - start:.1f}s")
    return StyleBank.from_client_templates(templates)


def evaluate_targets(cfg, model, targets, transform, device, amp) -> dict:
    return {name: evaluate_dataset(model, data, transform, cfg.eval, cfg.data, device, amp)
            for name, data in targets.items()}


def run(cfg: Config) -> dict:
    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "train.log")
    writer = MetricWriter(out_dir, cfg.log.tensorboard)
    seed_everything(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    amp = cfg.amp and device.type == "cuda"

    git = git_info(REPO_DIR)
    logger.info(f"git commit {git['commit']} dirty={git['dirty']} | torch {torch.__version__} | device {device}")
    logger.info("config:\n" + json.dumps(cfg.to_dict(), indent=2))
    dump_config(cfg, out_dir / "config.yaml")
    (out_dir / "git.json").write_text(json.dumps(git), encoding="utf-8")

    sources, targets = load_all_data(cfg, logger)

    need_clip = cfg.method.use_csa or cfg.model.backbone == "clip_vit"
    clip_model = tokenizer = None
    if need_clip:
        clip_model, tokenizer = build_clip(cfg.clip, (cfg.data.height, cfg.data.width))
        clip_model.requires_grad_(False)
    model, mean, std = build_image_encoder(cfg, clip_model)
    train_tf = build_train_transform(cfg.data, mean, std)
    test_tf = build_test_transform(cfg.data, mean, std)

    last = ckpt_dir / "last.pt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=False) \
        if cfg.ckpt.auto_resume and last.is_file() else None

    # ---- Phase I: purified anchors, cached per client (Alg. 1 lines 1-5)
    anchors = {}
    if cfg.method.use_csa:
        clip_model.to(device)
        for name, data in sources.items():
            anchors[name] = learn_anchors(cfg, data, clip_model, tokenizer, device, logger, writer)
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.to(device)
    clients = [Client(name, data, train_tf, model.feat_dim, cfg, device) for name, data in sources.items()]
    for client in clients:
        if client.name in anchors:
            client.set_anchors(anchors[client.name])

    # ---- GCSB: built once before the federated loop (Alg. 1 line 7)
    gsd = None
    if cfg.method.use_gsd:
        if checkpoint is not None:
            bank = StyleBank.from_state_dict(checkpoint["bank"])
        else:
            bank = build_style_bank(cfg, sources, test_tf, device, logger)
            atomic_save(bank.state_dict(), out_dir / "style_bank.pt")
        logger.info(f"[bank] {len(bank)} templates from {len(sources)} clients")
        gsd = GlobalStyleDiversification(bank, cfg.style.eps, cfg.style.per_image, cfg.seed)

    aggregator = build_aggregator(cfg.fed)
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    start_round, history = 0, []
    if checkpoint is not None:
        global_state = checkpoint["global_state"]
        for client in clients:
            client.classifier.load_state_dict(checkpoint["classifiers"][client.name])
        if gsd is not None:
            gsd.generator.set_state(checkpoint["gsd_generator"])
        set_rng_state(checkpoint["rng"])
        start_round, history = checkpoint["round"], checkpoint["history"]
        logger.info(f"resumed from {last} at round {start_round}")
        if checkpoint["config"] != cfg.to_dict():
            logger.warning("config differs from the one stored in the checkpoint being resumed")

    # ---- Coupled federated loop (Alg. 1 lines 8-16)
    for r in range(start_round, cfg.fed.rounds):
        tic = time.time()
        lr = round_lr(cfg.fed, r)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        states, scalars = [], {"lr": lr}
        for client in clients:
            state, stats = client.train_round(model, global_state, lr, gsd)
            states.append(state)
            scalars.update({f"{client.name}/{k}": v for k, v in stats.items()})
        global_state = aggregator.aggregate(states, [c.num_samples for c in clients])
        del states

        done = r + 1
        scalars["round_seconds"] = time.time() - tic
        if device.type == "cuda":
            scalars["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        writer.write("train", done, scalars)
        logger.info(f"[round {done}/{cfg.fed.rounds}] " + " ".join(f"{k} {v:.4g}" for k, v in scalars.items()))

        if done % cfg.eval.every == 0 or done == cfg.fed.rounds:
            model.load_state_dict(global_state)
            for name, res in evaluate_targets(cfg, model, targets, test_tf, device, amp).items():
                history.append({"round": done, "target": name, **res})
                writer.write("eval", done, {f"{name}/mAP": res["mAP"], f"{name}/R1": res["R1"]})
                logger.info(f"[eval round {done}] {name}: mAP {res['mAP']:.2%} R1 {res['R1']:.2%} "
                            f"R5 {res['R5']:.2%} R10 {res['R10']:.2%}")

        ckpt = {
            "round": done, "global_state": global_state, "history": history,
            "classifiers": {c.name: c.classifier.state_dict() for c in clients},
            "bank": gsd.bank.state_dict() if gsd is not None else None,
            "gsd_generator": gsd.generator.get_state() if gsd is not None else None,
            "rng": rng_state(), "config": cfg.to_dict(), "git": git,
        }
        atomic_save(ckpt, last)
        if done % cfg.ckpt.keep_every == 0 or done == cfg.fed.rounds:
            atomic_save(ckpt, ckpt_dir / f"round_{done:03d}.pt")

    results = summarize(history, cfg.fed.rounds)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, res in results.items():
        logger.info(f"[result] {name}: final mAP {res['final']['mAP']:.2%} R1 {res['final']['R1']:.2%} | "
                    f"best-evaluated round {res['best']['round']}: mAP {res['best']['mAP']:.2%} "
                    f"R1 {res['best']['R1']:.2%}")
    writer.close()
    return results


def summarize(history: list[dict], final_round: int) -> dict:
    """Per target: metrics at the final round and at the evaluated round with the best mAP."""
    out = {}
    for name in sorted({h["target"] for h in history}):
        rows = [h for h in history if h["target"] == name]
        final = next(h for h in rows if h["round"] == final_round)
        out[name] = {"final": final, "best": max(rows, key=lambda h: h["mAP"]), "all": rows}
    return out
