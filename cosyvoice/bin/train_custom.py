# Clean CosyVoice training script with native WandB, step-based logging,
# early stopping, and checkpoint management.
# Replaces cosyvoice/bin/train.py + executor.py patching approach.

from __future__ import print_function
import argparse
import datetime
import logging
import os
import re
import gc
import glob
import shutil
import torch
import torch.distributed as dist
from contextlib import nullcontext
from copy import deepcopy

logging.getLogger('matplotlib').setLevel(logging.WARNING)

from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config,
    batch_forward, batch_backward, update_parameter_and_lr,
    cosyvoice_join,
)

# Optional WandB
try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False


# ==========================================
# 📊 LOGGING & CHECKPOINT MANAGEMENT
# ==========================================

def hf_prefix(tag):
    """Map CosyVoice tags to HuggingFace Trainer-style prefixes."""
    return 'eval' if tag == 'CV' else 'train'


def log_metrics(tag, step, epoch, loss_dict, info_dict, rank):
    """Log metrics to console + WandB (HF Trainer style)."""
    p = hf_prefix(tag)
    lr = info_dict.get('lr', 0)
    grad_norm = info_dict.get('grad_norm', 0)

    # Console log
    loss_str = ' '.join([f'{k} {v:.6f}' for k, v in loss_dict.items()])
    if tag == 'TRAIN':
        logging.info(f'{tag} Step {step + 1} Epoch {epoch:.4f} | {loss_str} | lr {lr:.8f} grad_norm {grad_norm:.6f} | rank {rank}')
    else:
        logging.info(f'{tag} Step {step + 1} Epoch {epoch:.4f} | {loss_str} | lr {lr:.8f} | rank {rank}')

    # WandB log
    if _wandb_available and rank == 0 and wandb.run is not None:
        wandb_log = {
            f'{p}/epoch': epoch,
            f'{p}/global_step': step + 1,
            f'{p}/learning_rate': lr,
        }
        if tag == 'TRAIN' and 'grad_norm' in info_dict:
            wandb_log[f'{p}/grad_norm'] = grad_norm
        for k, v in loss_dict.items():
            wandb_log[f'{p}/{k}'] = v
        wandb.log(wandb_log, step=step + 1)


def cleanup_checkpoints(model_dir, keep_n=2):
    """Keep only the N most recent checkpoints + best_model.pt."""
    pts = sorted(glob.glob(os.path.join(model_dir, '*.pt')), key=os.path.getmtime)
    # Never delete best_model.pt or init.pt
    protected = {'best_model.pt', 'init.pt'}
    pts = [p for p in pts if os.path.basename(p) not in protected]

    while len(pts) > keep_n:
        old = pts.pop(0)
        os.remove(old)
        yaml_f = re.sub(r'\.pt$', '.yaml', old)
        if os.path.exists(yaml_f):
            os.remove(yaml_f)
        logging.info(f'[Checkpoint] 🗑️ Deleted old: {os.path.basename(old)}')


# ==========================================
# 📈 EVALUATION
# ==========================================

@torch.inference_mode()
def evaluate(model, cv_data_loader, info_dict, step, epoch, writer, rank):
    """Run cross-validation and return average loss."""
    logging.info(f'--- Eval at Step {step + 1} Epoch {epoch:.4f} ---')
    model.eval()

    total_num_utts = 0
    total_loss_dict = {}

    for batch_idx, batch_dict in enumerate(cv_data_loader):
        info_dict["tag"] = "CV"
        info_dict["step"] = step
        info_dict["epoch"] = epoch
        info_dict["batch_idx"] = batch_idx

        num_utts = len(batch_dict["utts"])
        total_num_utts += num_utts

        info_dict = batch_forward(model, batch_dict, None, info_dict)

        for k, v in info_dict['loss_dict'].items():
            if k not in total_loss_dict:
                total_loss_dict[k] = []
            total_loss_dict[k].append(v.mean().item() * num_utts)

    # Average losses
    for k, v in total_loss_dict.items():
        total_loss_dict[k] = sum(v) / total_num_utts

    # Log eval metrics
    info_dict['loss_dict'] = total_loss_dict
    log_metrics('CV', step, epoch, total_loss_dict, info_dict, rank)

    # TensorBoard
    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar(f'CV/{k}', info_dict[k], step + 1)
        for k, v in total_loss_dict.items():
            writer.add_scalar(f'CV/{k}', v, step + 1)

    model.train()

    # Return scalar loss for early stopping
    if 'loss' in total_loss_dict:
        return total_loss_dict['loss']
    elif total_loss_dict:
        return sum(total_loss_dict.values()) / len(total_loss_dict)
    return None


# ==========================================
# 🚂 TRAINING LOOP
# ==========================================

def train(model, optimizer, scheduler, scaler, train_data_loader, cv_data_loader,
          train_dataset, writer, info_dict, start_step, start_epoch, rank):
    """Clean step-based training loop with eval, early stopping, and checkpoint management."""

    # Config from env vars (set by kaggle_train.py)
    max_steps = int(os.environ.get('MAX_STEPS', '0'))
    log_interval = int(os.environ.get('LOG_INTERVAL', '5'))
    save_per_step = int(os.environ.get('SAVE_PER_STEP', '50'))
    keep_checkpoints = int(os.environ.get('KEEP_CHECKPOINTS', '2'))
    patience = int(os.environ.get('EARLY_STOPPING_PATIENCE', '0'))

    # State
    global_step = start_step
    best_eval_loss = float('inf')
    patience_counter = 0
    model_dir = info_dict['model_dir']

    logging.info(f'=== Training Config ===')
    logging.info(f'  max_steps:        {max_steps}')
    logging.info(f'  log_interval:     {log_interval} steps')
    logging.info(f'  save_per_step:    {save_per_step} steps')
    logging.info(f'  keep_checkpoints: {keep_checkpoints}')
    logging.info(f'  patience:         {patience}')
    logging.info(f'  starting step:    {start_step}')
    logging.info(f'========================')

    stop_training = False

    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        if stop_training:
            break

        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=info_dict.get('timeout', 60)))

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} | lr {lr} | rank {rank}')

        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext

        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = global_step
                info_dict["epoch"] = epoch
                info_dict["batch_idx"] = batch_idx

                if cosyvoice_join(group_join, info_dict):
                    break

                # Gradient accumulation context
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)

                # Check if this is an accumulation boundary (optimizer step)
                is_accum_boundary = (
                    (info_dict['train_engine'] == 'deepspeed' and info_dict.get('is_gradient_accumulation_boundary', False)) or
                    (info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict['accum_grad'] == 0)
                )

                if is_accum_boundary:
                    global_step += 1
                    
                    # Calculate fractional epoch (like HF Trainer)
                    try:
                        fractional_epoch = epoch + (batch_idx + 1) / len(train_data_loader)
                    except TypeError:  # Handle IterableDataset missing len() safely
                        fractional_epoch = float(epoch)

                    # --- LOG ---
                    if global_step % log_interval == 0:
                        log_metrics('TRAIN', global_step - 1, fractional_epoch, info_dict['loss_dict'], info_dict, rank)

                    # --- EVAL + SAVE ---
                    if save_per_step > 0 and global_step % save_per_step == 0:
                        dist.barrier()
                        eval_loss = evaluate(model, cv_data_loader, info_dict, global_step - 1, fractional_epoch, writer, rank)

                        # Save checkpoint
                        info_dict['step'] = global_step - 1
                        model_name = f'step_{global_step}'
                        save_model(model, model_name, info_dict)

                        # Best model tracking
                        if eval_loss is not None:
                            if eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                patience_counter = 0
                                save_model(model, 'best_model', info_dict)
                                logging.info(f'[Best] 🏆 New best eval loss: {eval_loss:.6f}')
                            else:
                                patience_counter += 1
                                logging.info(f'[EarlyStopping] No improvement {patience_counter}/{patience} '
                                             f'(best={best_eval_loss:.6f}, current={eval_loss:.6f})')

                            if patience > 0 and patience_counter >= patience:
                                logging.info(f'[EarlyStopping] ⛔ Stopping after {patience} evals without improvement')
                                stop_training = True

                        # Checkpoint cleanup
                        if rank == 0:
                            cleanup_checkpoints(model_dir, keep_n=keep_checkpoints)

                        model.train()

                    # --- MAX STEPS ---
                    if max_steps > 0 and global_step >= max_steps:
                        logging.info(f'[MaxSteps] ✅ Reached {global_step}/{max_steps} steps. Stopping.')
                        stop_training = True

                    if stop_training:
                        break

        dist.destroy_process_group(group_join)

        # No forced end-of-epoch eval. Evaluation is strictly step-based.

    # Clean up WandB
    if _wandb_available and rank == 0 and wandb.run is not None:
        wandb.finish()

    logging.info(f'Training complete! Total steps: {global_step}, Best eval loss: {best_eval_loss:.6f}')


# ==========================================
# 🚀 MAIN
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description='CosyVoice Custom Training')
    parser.add_argument('--train_engine', default='torch_ddp', choices=['torch_ddp', 'deepspeed'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--cv_data', required=True)
    parser.add_argument('--qwen_pretrain_path', required=False)
    parser.add_argument('--onnx_path', required=False)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--tensorboard_dir', default='tensorboard')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--prefetch', default=100, type=int)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--dpo', action='store_true', default=False)
    parser.add_argument('--deepspeed.save_states', dest='save_states', default='model_only')
    parser.add_argument('--timeout', default=60, type=int)

    import deepspeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    os.environ['onnx_path'] = args.onnx_path or ''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    rank = int(os.environ.get('RANK', 0))
    gan = args.model == 'hifigan'

    # --- Fix prefetch_factor for num_workers=0 (PyTorch 2.3.1 compat) ---
    if args.num_workers == 0:
        args.prefetch = None

    # --- Load config ---
    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan:
        override_dict.pop('hift', None)
    if args.qwen_pretrain_path:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))

    # --- Init DDP ---
    init_distributed(args)

    # --- Dataset ---
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)

    configs = check_modify_and_save_config(args, configs)
    writer = init_summarywriter(args)

    # --- Init WandB (rank 0 only) ---
    if rank == 0 and os.environ.get('WANDB_PROJECT'):
        try:
            wandb.init(
                project=os.environ['WANDB_PROJECT'],
                name=os.environ.get('WANDB_RUN_NAME', 'cosyvoice-train'),
                config={k: v for k, v in configs.get('train_conf', {}).items() if isinstance(v, (int, float, str, bool))}
            )
            logging.info('WandB initialized ✅')
        except Exception as e:
            logging.warning(f'WandB init failed: {e}')

    # --- Load model + checkpoint ---
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        start_step = state_dict.get('step', 0)
        start_epoch = state_dict.get('epoch', -1)
        logging.info(f'Loaded checkpoint: step={start_step}, epoch={start_epoch}')

    # --- Wrap model ---
    model = wrap_cuda_model(args, model)

    # --- Optimizer & scheduler ---
    model, optimizer, scheduler, _, _ = init_optimizer_and_scheduler(args, configs, model, gan)
    scheduler.set_step(start_step)

    # --- Save init checkpoint ---
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # --- Scaler for AMP ---
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    logging.info(f'Starting training from step {start_step}, epoch {start_epoch}')

    # --- Run training ---
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_data_loader=train_data_loader,
        cv_data_loader=cv_data_loader,
        train_dataset=train_dataset,
        writer=writer,
        info_dict=info_dict,
        start_step=start_step,
        start_epoch=start_epoch,
        rank=rank,
    )

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
