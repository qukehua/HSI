import os
import sys
import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault('ROOT_DIR', str(PROJECT_ROOT))
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from utils import *
from constants import *
from torch.utils.tensorboard import SummaryWriter
from datasets.lingo import LingoDataset
from tqdm.auto import tqdm


def get_split_subset(dataset, dataset_cfg, split_name):
    if split_name in [None, 'None', 'none', 'null']:
        return dataset
    split_dir = dataset_cfg.get('split_dir', 'splits')
    split_path = os.path.join(dataset_cfg.folder, split_dir, f'{split_name}_idx.npy')
    split_idx = np.load(split_path).astype(np.int64)
    return Subset(dataset, split_idx)


def move_lingo_batch(batch, device):
    joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, length, valid_mask, object_present = batch
    return joints.to(device), \
           mat.to(device), scene_flag.to(device), \
           text_clip_embedding.to(device), \
           pelvis_goal.to(device), hand_goal.to(device), \
           is_pick.to(device), need_scene.to(device), need_pelvis_dir.to(device), pi.to(device), \
           need_pi.to(device), is_loco.to(device), length.to(device), valid_mask.to(device), object_present.to(device)


@torch.no_grad()
def validate(model, trainer, dataloader, cfg, device, epoch=0, show_progress=False):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)

    progress = tqdm(dataloader, desc=f"Val {epoch}", disable=not show_progress, leave=False)
    for batch in progress:
        joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, length, valid_mask, object_present = move_lingo_batch(batch, device)
        batch_size = joints.shape[0]
        t = torch.randint(0, trainer.timesteps, (batch_size,), device=device).long()
        mask, _, _ = get_mask(joints, -1, p=1., fixed_frame=cfg.auto_regre_num)
        loss = trainer.p_losses(
            joints, mat, scene_flag, mask, t,
            text_clip_embedding, pelvis_goal, hand_goal,
            is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco,
            length=length, valid_mask=valid_mask, object_present=object_present,
        )
        total_loss += loss.detach() * batch_size
        total_count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_count, op=torch.distributed.ReduceOp.SUM)
    model.train()
    return (total_loss / total_count.clamp_min(1)).item()


@hydra.main(version_base=None, config_path="config", config_name="config_train_model")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = cfg.num_gpus
    print('Usable GPUS: ', torch.cuda.device_count(), flush=True)
    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, cfg),
                                nprocs=world_size,
                                join=True)

def train_ddp(rank, world_size, cfg):

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)

    train_split = cfg.dataset.get('split', 'train')
    val_split = cfg.get('val_split', 'val')
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_cfg.split = None
    synhsi_dataset = LingoDataset(**dataset_cfg)
    train_dataset = get_split_subset(synhsi_dataset, dataset_cfg, train_split)
    val_dataset = get_split_subset(synhsi_dataset, dataset_cfg, val_split) if cfg.get('use_validation', True) else None

    sampler = DistributedSampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=True)
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, drop_last=True, num_workers=cfg.num_workers,
                                    sampler=val_sampler, pin_memory=True)
    else:
        val_dataloader = None

    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    trainer.set_dataset_and_model(synhsi_dataset, model)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))

    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        if rank == 0:
            print(f'Start epoch {epoch}', flush=True)
        sampler.set_epoch(epoch)
        step = 0
        progress = tqdm(dataloader, desc=f"Train {epoch}", disable=rank != 0, leave=True)
        for batch in progress:
            step += 1
            optimizer.zero_grad()

            joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, length, valid_mask, object_present = move_lingo_batch(batch, device)

            batch_size = joints.shape[0]
            t = torch.randint(0, trainer.timesteps, (batch_size,), device=device).long()
            with torch.no_grad():
                mask, _, _ = get_mask(joints, -1, p=1., fixed_frame=cfg.auto_regre_num)

            loss = trainer.p_losses(
                joints, mat, scene_flag, mask, t,
                text_clip_embedding, pelvis_goal, hand_goal,
                is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco,
                length=length, valid_mask=valid_mask, object_present=object_present,
            )

            if step % 10 == 0:
                print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}", flush=True)
                if cfg.use_tensorboard and rank == 0:
                    writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)

            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            print(f'Saving checkpoint', flush=True)
            ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"))

        if val_dataloader is not None and epoch % cfg.val_interval == 0:
            val_loss = validate(model, trainer, val_dataloader, cfg, device, epoch=epoch, show_progress=(rank == 0))
            if rank == 0:
                print(f"Epoch: {epoch}   Val Loss: {val_loss}", flush=True)
                if cfg.use_tensorboard:
                    writer.add_scalar('Val/Loss', val_loss, epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
                    os.makedirs(ckpt_folder, exist_ok=True)
                    best_path = os.path.join(ckpt_folder, 'best_model.pth')
                    torch.save(model.module.state_dict(), best_path)
                    print(f"Saved best model to {best_path} with val loss {best_val_loss}", flush=True)

        torch.distributed.barrier()

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()


def get_mask(x_start, ind, p, fixed_frame=0, mask_y=True):
    '''
    get mask for the input sequence of pre frames and final goal frame
    '''
    mask_frame = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)
    mask_goal = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)

    # goal mask
    if ind != -1:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_goal[rand_batch, -1, ind * 3: ind * 3 + 3] = True
        if not mask_y:
            mask_goal[rand_batch, -1, ind * 3 + 1] = False

    # prefix frame mask
    if fixed_frame > 0:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_frame[rand_batch, :fixed_frame, :] = True
    mask = torch.logical_or(mask_frame, mask_goal)
    return mask, mask_frame, mask_goal


if __name__ == '__main__':
    train()
