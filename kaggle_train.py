import os, sys, json, shutil, subprocess, tempfile, logging
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ==========================================
# 🎛️ TRAINING CONFIGURATION
# ==========================================

# --- Model Paths ---
PRETRAINED_MODEL_DIR = "/kaggle/input/datasets/rustedpipe/cosyvoice3-pretrained/Fun-CosyVoice3-0.5B"  # Kaggle dataset input
COSYVOICE_ROOT = "/kaggle/working/CosyVoice"           # CosyVoice repository root

# --- What to Train ---
TRAIN_LLM = True           # Required: always finetune the LLM
TRAIN_FLOW = False          # Optional: also finetune the Flow (diffusion) stage

# --- Training Hyperparams ---
LEARNING_RATE = 1e-5
MAX_STEPS = 1000            # Stop training after N optimizer steps
MAX_EPOCH = 9999             # Set very high; MAX_STEPS controls actual end
ACCUM_GRAD = 2
GRAD_CLIP = 5
MAX_FRAMES_IN_BATCH = 1000   # Controls batch size (decrease if OOM, increase for speed)
SAVE_PER_STEP = 50           # Save checkpoint + run eval every N optimizer steps (-1 = epoch end only)
LOG_INTERVAL = 5         # Log loss to console + WandB every N batches
WARMUP_STEPS = 0             # Number of warmup steps for learning rate scheduler

# --- Checkpoint Management ---
KEEP_CHECKPOINTS = 2         # Keep only the N most recent checkpoints (+ best)

# --- Early Stopping ---
EARLY_STOPPING = True        # Enable early stopping based on eval loss
EARLY_STOPPING_PATIENCE = 5  # Stop after N evals without eval loss improvement

# --- Data Paths ---
TRAIN_DATA_LIST = "/kaggle/working/data/train/data.list"
CV_DATA_LIST = "/kaggle/working/data/validation/data.list"

# --- DeepSpeed ---
DS_STAGE = 2
DS_OFFLOAD_OPTIMIZER = "none"    # "none" or "cpu"
DS_BF16 = False                  # Use bf16 via DeepSpeed

# --- Output ---
OUTPUT_DIR = "/kaggle/working/output"
NUM_GPUS = 1  # Use 1 GPU to avoid DDP + gradient checkpointing + LoRA conflict

# --- Weights & Biases ---
USE_WANDB = True
WANDB_PROJECT = "huggingface"   # WandB project name
WANDB_RUN_NAME = "cosyvoice-full-finetune-test"  # WandB run name
WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")

# ==========================================
# 🔧 INTERNAL: Generate temporary config files
# ==========================================

def generate_deepspeed_config(tmp_dir):
    """Generate a DeepSpeed JSON config from inline settings."""
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": ACCUM_GRAD,
        "steps_per_print": LOG_INTERVAL,
        "gradient_clipping": GRAD_CLIP,
        "fp16": {
            "enabled": False,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 256,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": DS_BF16
        },
        "zero_force_ds_cpu_optimizer": False,
        "zero_optimization": {
            "stage": DS_STAGE,
            "offload_optimizer": {
                "device": DS_OFFLOAD_OPTIMIZER,
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },
        "checkpoint": {
            "save_optimizer_states": False,
            "save_lr_scheduler_states": False
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LEARNING_RATE,
                "weight_decay": 0.0001,
                "torch_adam": True,
                "adam_w_mode": True
            }
        }
    }
    ds_path = os.path.join(tmp_dir, "ds_config.json")
    with open(ds_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    return ds_path


def patch_yaml_train_conf(yaml_path, tmp_dir):
    """Patch train_conf hyperparams in the YAML (simple text replacement)."""
    with open(yaml_path, 'r') as f:
        content = f.read()

    # Override train_conf values via text replacement
    import re
    content = re.sub(r'(train_conf:\s*\n\s*optim:\s*)\w+', r'\1adam', content)
    content = re.sub(r'lr:\s*[\d.e\-]+\s*#.*change.*sft', f'lr: {LEARNING_RATE}', content)
    content = re.sub(r'scheduler:\s*constantlr.*sft', 'scheduler: warmuplr', content)
    content = re.sub(r'warmup_steps:\s*\d+', f'warmup_steps: {WARMUP_STEPS}', content)
    content = re.sub(r'max_epoch:\s*\d+', f'max_epoch: {MAX_EPOCH}', content)
    content = re.sub(r'accum_grad:\s*\d+', f'accum_grad: {ACCUM_GRAD}', content)
    content = re.sub(r'grad_clip:\s*\d+', f'grad_clip: {GRAD_CLIP}', content)
    content = re.sub(r'max_frames_in_batch:\s*\d+', f'max_frames_in_batch: {MAX_FRAMES_IN_BATCH}', content)
    content = re.sub(r'save_per_step:\s*-?\d+', f'save_per_step: {SAVE_PER_STEP}', content)
    content = re.sub(r'log_interval:\s*\d+', f'log_interval: {LOG_INTERVAL}', content)

    patched_path = os.path.join(tmp_dir, "cosyvoice3_train.yaml")
    with open(patched_path, 'w') as f:
        f.write(content)
    return patched_path

def patch_prefetch_factor():
    """Fix prefetch_factor=None when num_workers=0 (PyTorch 2.3.1 compat).
       This is the ONLY patch needed — train_custom.py handles everything else.
    """
    train_utils_path = os.path.join(COSYVOICE_ROOT, "cosyvoice", "utils", "train_utils.py")
    with open(train_utils_path, 'r') as f:
        content = f.read()
    if "prefetch_factor=args.prefetch" in content and "if args.num_workers > 0" not in content:
        content = content.replace(
            "prefetch_factor=args.prefetch)",
            "prefetch_factor=args.prefetch if args.num_workers > 0 else None)"
        )
        with open(train_utils_path, 'w') as f:
            f.write(content)
        logging.info("Patched train_utils.py: prefetch_factor fix for num_workers=0")


def run_training(model_name, yaml_path, ds_path):
    """Launch train_custom.py via torchrun."""
    patch_prefetch_factor()

    train_script = os.path.join(COSYVOICE_ROOT, "cosyvoice", "bin", "train_custom.py")
    train_engine = "deepspeed"

    cmd = [
        "torchrun",
        f"--nnodes=1",
        f"--nproc_per_node={NUM_GPUS}",
        train_script,
        "--train_engine", train_engine,
        "--model", model_name,
        "--config", yaml_path,
        "--train_data", TRAIN_DATA_LIST,
        "--cv_data", CV_DATA_LIST,
        "--model_dir", os.path.join(OUTPUT_DIR, model_name),
        "--num_workers", "0",
    ]

    # DeepSpeed-specific args
    cmd.extend(["--deepspeed_config", ds_path])
    cmd.extend(["--deepspeed.save_states", "model_only"])

    # Add qwen_pretrain_path
    qwen_path = os.path.join(PRETRAINED_MODEL_DIR, "CosyVoice-BlankEN")
    if not os.path.isdir(qwen_path):
        qwen_path = PRETRAINED_MODEL_DIR
    cmd.extend(["--qwen_pretrain_path", qwen_path])

    # Add onnx path
    cmd.extend(["--onnx_path", PRETRAINED_MODEL_DIR])

    logging.info(f"Running: {' '.join(cmd)}")

    # Set PYTHONPATH so torchrun subprocess can find cosyvoice and matcha packages
    env = os.environ.copy()
    matcha_path = os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{COSYVOICE_ROOT}:{matcha_path}:{existing}"

    # Pass WandB env vars so the subprocess can log metrics
    if USE_WANDB and WANDB_API_KEY:
        env["WANDB_API_KEY"] = WANDB_API_KEY
        env["WANDB_PROJECT"] = WANDB_PROJECT
        env["WANDB_RUN_NAME"] = WANDB_RUN_NAME
        env["WANDB_MODE"] = "online"

    # Pass training loop config params
    if EARLY_STOPPING:
        env["EARLY_STOPPING_PATIENCE"] = str(EARLY_STOPPING_PATIENCE)
    env["MAX_STEPS"] = str(MAX_STEPS)
    env["KEEP_CHECKPOINTS"] = str(KEEP_CHECKPOINTS)
    env["SAVE_PER_STEP"] = str(SAVE_PER_STEP)
    env["LOG_INTERVAL"] = str(LOG_INTERVAL)

    result = subprocess.run(cmd, cwd=COSYVOICE_ROOT, env=env)
    if result.returncode != 0:
        logging.error(f"Training failed for model={model_name} with return code {result.returncode}")
        sys.exit(result.returncode)
    logging.info(f"Training completed for model={model_name}")


# ==========================================
# 🚀 MAIN
# ==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Weights & Biases (set env vars for subprocess; init happens in torchrun child)
    if USE_WANDB:
        if WANDB_API_KEY:
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT
        os.environ["WANDB_RUN_NAME"] = WANDB_RUN_NAME
        os.environ["WANDB_MODE"] = "online"
        logging.info(f"WandB configured: project={WANDB_PROJECT}, run={WANDB_RUN_NAME}")

    # Create temporary directory for generated configs
    tmp_dir = tempfile.mkdtemp(prefix="cosyvoice_train_")
    logging.info(f"Temporary config dir: {tmp_dir}")

    # Base YAML path
    base_yaml = os.path.join(
        COSYVOICE_ROOT, "examples", "libritts", "cosyvoice3", "conf", "cosyvoice3.yaml"
    )
    if not os.path.exists(base_yaml):
        logging.error(f"Base YAML not found: {base_yaml}")
        sys.exit(1)

    # 1. Generate DeepSpeed config
    ds_path = generate_deepspeed_config(tmp_dir)

    # 2. Patch YAML with training hyperparams
    yaml_path = patch_yaml_train_conf(base_yaml, tmp_dir)

    # 4. Run LLM training (required)
    if TRAIN_LLM:
        logging.info("=" * 50)
        logging.info("Stage 1: Training LLM")
        logging.info("=" * 50)
        run_training("llm", yaml_path, ds_path)

    # 5. Run Flow training (optional)
    if TRAIN_FLOW:
        logging.info("=" * 50)
        logging.info("Stage 2: Training Flow")
        logging.info("=" * 50)
        run_training("flow", yaml_path, ds_path)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info("All training stages complete!")


if __name__ == '__main__':
    main()
