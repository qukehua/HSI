import pdb

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from utils import *
from constants import *
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from datasets.scheduler import SchedulerDataset
from models.synhsi import TimingModel
from tqdm.auto import tqdm

os.environ['ROOT_DIR'] = '..'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import sys
sys.path.append(os.path.join(os.environ['ROOT_DIR'], 'code'))


def get_split_subset(dataset, dataset_cfg, split_name):
    if split_name in [None, 'None', 'none', 'null']:
        return dataset
    split_dir = dataset_cfg.get('split_dir', 'splits')
    split_path = os.path.join(dataset_cfg.folder, split_dir, f'scheduler_{split_name}_idx.npy')
    split_idx = np.load(split_path).astype(np.int64)
    return Subset(dataset, split_idx)


@torch.no_grad()
def validate_scheduler(model, dataloader, loss_fn, device, epoch=0, show_progress=True):
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0

    progress = tqdm(dataloader, desc=f"Val {epoch}", disable=not show_progress, leave=False)
    for batch in progress:
        joints, stop, pi, text_clip_embedding = batch
        joints = joints.to(device)
        stop = stop.to(device=device, dtype=torch.float32)
        pi = pi.to(device)
        text_clip_embedding = text_clip_embedding.to(device)

        stop_pred = model(joints, text_clip_embedding, pi).squeeze(1)
        loss = loss_fn(stop_pred, stop)
        batch_size = joints.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size
        correct += ((stop_pred > 0.5) == (stop > 0.5)).sum().item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    model.train()
    mean_loss = total_loss / max(total_count, 1)
    accuracy = correct / max(total_count, 1)
    return mean_loss, accuracy


@hydra.main(version_base=None, config_path="config", config_name="config_train_scheduler")
def train(cfg):
    rank = 0
    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)

    model = TimingModel(**cfg.model.scheduler)
    model.to(device)
    model.train()

    train_split = cfg.dataset.get('split', 'train')
    val_split = cfg.get('val_split', 'val')
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_cfg.split = None
    scheduler_dataset = SchedulerDataset(**dataset_cfg)
    train_dataset = get_split_subset(scheduler_dataset, dataset_cfg, train_split)
    val_dataset = get_split_subset(scheduler_dataset, dataset_cfg, val_split) if cfg.get('use_validation', True) else None

    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            shuffle=True, pin_memory=True)
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, drop_last=False, num_workers=cfg.num_workers,
                                    shuffle=False, pin_memory=True)
    else:
        val_dataloader = None

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))

    loss_fn = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):

        print(f'Start epoch {epoch}', flush=True)
        step = 0
        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            print(f'Saving checkpoint', flush=True)
            ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"))

        progress = tqdm(dataloader, desc=f"Train {epoch}", leave=True)
        for batch in progress:
            step += 1

            optimizer.zero_grad()

            joints, stop, pi, text_clip_embedding = batch
            joints, stop, pi, text_clip_embedding = joints.to(device), stop.to(device=device, dtype=torch.float32), pi.to(device), text_clip_embedding.to(device)

            stop_pred = model(joints, text_clip_embedding, pi).squeeze(1)

            loss = loss_fn(stop_pred, stop)

            if step % 10 == 1:
                print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}", flush=True)
                if cfg.use_tensorboard and rank == 0:
                    writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)

                with torch.no_grad():
                    thres = 0.5
                    pred = stop_pred > thres
                    acc_0 = (pred[stop == 0] == stop[stop == 0]).float().mean().item()
                    acc_1 = (pred[torch.logical_not(stop==0)] == stop[torch.logical_not(stop==0)]).float().mean().item()
                    print(f"Accuracy on class 0: {acc_0}, class 1: {acc_1}", flush=True)

            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if val_dataloader is not None and epoch % cfg.val_interval == 0:
            val_loss, val_acc = validate_scheduler(model, val_dataloader, loss_fn, device, epoch=epoch, show_progress=True)
            print(f"Epoch: {epoch}   Val Loss: {val_loss}   Val Acc: {val_acc}", flush=True)
            if cfg.use_tensorboard and rank == 0:
                writer.add_scalar('Val/Loss', val_loss, epoch)
                writer.add_scalar('Val/Accuracy', val_acc, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
                os.makedirs(ckpt_folder, exist_ok=True)
                best_path = os.path.join(ckpt_folder, 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                print(f"Saved best scheduler model to {best_path} with val loss {best_val_loss}", flush=True)

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
