import datetime
import json
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("ROOT_DIR", str(PROJECT_ROOT))
os.environ["CURRENT_TIME"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.motion_evaluator import MotionEvaluator
from train import build_dataset, get_split_subset, move_lingo_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def denormalize_if_needed(dataset, joints: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and hasattr(dataset, "denormalize_torch"):
        return dataset.denormalize_torch(joints)
    return joints


def infer_dims(dataset) -> tuple[int, int]:
    sample = dataset[0]
    motion = np.asarray(sample[0])
    text = np.asarray(sample[3])
    return int(motion.shape[-1]), int(text.shape[-1])


def build_loaders(cfg: DictConfig):
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_cfg.split = None
    dataset_cfg.train = True
    dataset_cfg.load_language = True
    dataset_cfg.load_scene = False
    dataset_cfg.load_hand_goal = False
    dataset_cfg.load_pelvis_goal = False
    dataset = build_dataset(dataset_cfg)

    train_dataset = get_split_subset(dataset, dataset_cfg, cfg.train_split)
    val_dataset = get_split_subset(dataset, dataset_cfg, cfg.val_split)
    pin_memory = int(cfg.num_workers) > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    return dataset, train_loader, val_loader


def move_batch_for_evaluator(batch, device: torch.device, dataset, denormalize_motion: bool):
    joints, _, _, text_emb, _, _, _, _, _, _, _, _, length, valid_mask, _, _, _ = move_lingo_batch(batch, device)
    joints = denormalize_if_needed(dataset, joints, denormalize_motion)
    length = length.to(device=device, dtype=torch.long)
    if valid_mask is not None:
        valid_len = valid_mask.to(device=device, dtype=torch.bool).sum(dim=1).to(torch.long)
        length = torch.minimum(length, valid_len.clamp_min(1))
    return joints.float(), text_emb.float(), length


def average_meter(total: dict, count: int) -> dict:
    if count == 0:
        return {}
    return {key: value / count for key, value in total.items()}


def add_logs(total: dict, logs: dict, batch_size: int) -> None:
    for key, value in logs.items():
        if torch.is_tensor(value):
            value = float(value.detach().cpu().item())
        total[key] = total.get(key, 0.0) + float(value) * batch_size


def run_epoch(model, loader, dataset, optimizer, cfg, device: torch.device, train: bool, epoch: int):
    model.train(mode=train)
    totals = {}
    count = 0
    desc = f"{'Train' if train else 'Val'} {epoch}"
    progress = tqdm(loader, desc=desc, leave=train, disable=not cfg.show_progress)
    for batch in progress:
        joints, text_emb, length = move_batch_for_evaluator(batch, device, dataset, cfg.denormalize_motion)
        batch_size = joints.shape[0]
        if batch_size < 2 and train:
            continue

        if train:
            optimizer.zero_grad(set_to_none=True)
            logs = model.contrastive_loss(joints, text_emb, length)
            logs["loss"].backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                logs = model.contrastive_loss(joints, text_emb, length)

        add_logs(totals, logs, batch_size)
        count += batch_size
        progress.set_postfix(
            loss=f"{float(logs['loss'].detach().cpu()):.4f}",
            top1=f"{float(logs['top1_t2m'].detach().cpu()):.3f}",
        )
    return average_meter(totals, count)


def save_checkpoint(path: Path, model: MotionEvaluator, optimizer, cfg, epoch: int, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "config": model.config,
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "dataset_name": cfg.dataset.get("name", None),
            "denormalize_motion": bool(cfg.denormalize_motion),
        },
        path,
    )


@hydra.main(version_base=None, config_path="config", config_name="config_motion_evaluator_lingo")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    cfg.device = str(device)
    os.makedirs(cfg.exp_dir, exist_ok=True)

    dataset, train_loader, val_loader = build_loaders(cfg)
    motion_dim, text_dim = infer_dims(dataset)
    model = MotionEvaluator(
        motion_dim=motion_dim,
        text_dim=text_dim,
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        root_joint=cfg.root_joint,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    metrics_path = Path(cfg.exp_dir) / "motion_evaluator_metrics.jsonl"
    best_loss = float("inf")

    for epoch in range(int(cfg.epochs)):
        train_logs = run_epoch(model, train_loader, dataset, optimizer, cfg, device, True, epoch)
        val_logs = run_epoch(model, val_loader, dataset, optimizer, cfg, device, False, epoch)
        val_loss = val_logs.get("loss", float("inf"))
        payload = {
            "epoch": epoch,
            **{f"train/{key}": value for key, value in train_logs.items()},
            **{f"val/{key}": value for key, value in val_logs.items()},
        }
        print(json.dumps(payload, indent=2))
        write_jsonl(metrics_path, payload)

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(Path(cfg.exp_dir) / "checkpoints" / "best_model.pth", model, optimizer, cfg, epoch, val_loss)
            print(f"Saved best evaluator with val loss {best_loss:.6f}")

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                Path(cfg.exp_dir) / "checkpoints" / f"motion_evaluator_epoch{epoch:03d}.pth",
                model,
                optimizer,
                cfg,
                epoch,
                val_loss,
            )

    save_checkpoint(Path(cfg.exp_dir) / "checkpoints" / "latest_model.pth", model, optimizer, cfg, int(cfg.epochs) - 1, best_loss)


if __name__ == "__main__":
    main()
