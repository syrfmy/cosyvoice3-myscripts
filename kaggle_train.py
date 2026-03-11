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

# --- QLoRA Configuration ---
USE_QLORA = True            # Enable QLoRA for the LLM's Qwen2 backbone
QLORA_MODE = "attn+mlp"     # "attn_only" or "attn+mlp"
LORA_R = 64                 # LoRA rank
LORA_ALPHA = 128            # LoRA alpha
LORA_DROPOUT = 0.05         # LoRA dropout

# --- Training Hyperparams ---
LEARNING_RATE = 1e-5
MAX_STEPS = 1000            # Stop training after N optimizer steps
MAX_EPOCH = 9999             # Set very high; MAX_STEPS controls actual end
ACCUM_GRAD = 2
GRAD_CLIP = 5
SAVE_PER_STEP = 50           # Save checkpoint + run eval every N optimizer steps (-1 = epoch end only)
LOG_INTERVAL = 5         # Log loss to console + WandB every N batches

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
WANDB_RUN_NAME = "cosyvoice-test-run"  # WandB run name
WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")

# ==========================================
# 🔧 INTERNAL: Generate temporary config files
# ==========================================

def get_qlora_target_modules():
    """Return LoRA target modules based on QLORA_MODE."""
    if QLORA_MODE == "attn_only":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif QLORA_MODE == "attn+mlp":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        raise ValueError(f"Unknown QLORA_MODE: {QLORA_MODE}. Use 'attn_only' or 'attn+mlp'.")


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


def patch_yaml_for_qlora(yaml_path, tmp_dir):
    """Read the original cosyvoice3.yaml, replace llm class with llm_lora, write to tmp."""
    with open(yaml_path, 'r') as f:
        content = f.read()

    # Replace the LLM class to use llm_lora module
    content = content.replace(
        "!new:cosyvoice.llm.llm.CosyVoice3LM",
        "!new:cosyvoice.llm.llm_lora.CosyVoice3LM"
    )
    content = content.replace(
        "!new:cosyvoice.llm.llm.Qwen2Encoder",
        "!new:cosyvoice.llm.llm_lora.Qwen2Encoder"
    )

    patched_path = os.path.join(tmp_dir, "cosyvoice3_qlora.yaml")
    with open(patched_path, 'w') as f:
        f.write(content)
    return patched_path


def patch_yaml_train_conf(yaml_path, tmp_dir):
    """Patch train_conf hyperparams in the YAML (simple text replacement)."""
    with open(yaml_path, 'r') as f:
        content = f.read()

    # Override train_conf values via text replacement
    import re
    content = re.sub(r'(train_conf:\s*\n\s*optim:\s*)\w+', r'\1adam', content)
    content = re.sub(r'lr:\s*[\d.e\-]+\s*#.*change.*sft', f'lr: {LEARNING_RATE}', content)
    content = re.sub(r'max_epoch:\s*\d+', f'max_epoch: {MAX_EPOCH}', content)
    content = re.sub(r'accum_grad:\s*\d+', f'accum_grad: {ACCUM_GRAD}', content)
    content = re.sub(r'grad_clip:\s*\d+', f'grad_clip: {GRAD_CLIP}', content)
    content = re.sub(r'save_per_step:\s*-?\d+', f'save_per_step: {SAVE_PER_STEP}', content)
    content = re.sub(r'log_interval:\s*\d+', f'log_interval: {LOG_INTERVAL}', content)

    patched_path = os.path.join(tmp_dir, "cosyvoice3_train.yaml")
    with open(patched_path, 'w') as f:
        f.write(content)
    return patched_path


def ensure_llm_lora_exists():
    """Ensure llm_lora.py exists in cosyvoice/llm/. If not, create it."""
    lora_path = os.path.join(COSYVOICE_ROOT, "cosyvoice", "llm", "llm_lora.py")
    # Always regenerate to ensure latest patches are applied

    logging.info(f"Creating llm_lora.py at: {lora_path}")
    target_modules = get_qlora_target_modules()

    # Read the original llm.py and inject QLoRA
    orig_path = os.path.join(COSYVOICE_ROOT, "cosyvoice", "llm", "llm.py")
    with open(orig_path, 'r') as f:
        content = f.read()

    # Add peft/bnb imports
    content = content.replace(
        "from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path",
        "from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path\n"
        "\nfrom peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
        "from transformers import BitsAndBytesConfig"
    )

    # Replace Qwen2Encoder __init__
    old_init = '''class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)'''

    new_init = f'''class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = Qwen2ForCausalLM.from_pretrained(
            pretrain_path,
            quantization_config=bnb_config,
            device_map={{"": local_rank}},
            torch_dtype=torch.float16
        )

        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r={LORA_R},
            lora_alpha={LORA_ALPHA},
            target_modules={target_modules},
            lora_dropout={LORA_DROPOUT},
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        if local_rank == 0:
            self.model.print_trainable_parameters()'''

    content = content.replace(old_init, new_init)

    # Fix attribute path for PeftModel wrapper:
    # Original: self.llm.model.model.embed_tokens → Qwen2Encoder.model(Qwen2ForCausalLM).model(Qwen2Model).embed_tokens
    # With PeftModel: self.llm.model(PeftModel).model(Qwen2ForCausalLM).model(Qwen2Model).embed_tokens
    content = content.replace(
        "self.llm.model.model.embed_tokens",
        "self.llm.model.model.model.embed_tokens"
    )

    # Also fix forward_one_step if it accesses lm_head
    content = content.replace(
        "self.llm.model.lm_head",
        "self.llm.model.model.lm_head"
    )

    # Fix gradient checkpointing to use non-reentrant mode (avoids DDP conflict)
    content = content.replace(
        "self.model = prepare_model_for_kbit_training(self.model)",
        "self.model = prepare_model_for_kbit_training(self.model)\n"
        "        self.model.gradient_checkpointing_enable(\n"
        "            gradient_checkpointing_kwargs={'use_reentrant': False}\n"
        "        )"
    )

    with open(lora_path, 'w') as f:
        f.write(content)
    logging.info(f"Created llm_lora.py with QLoRA targeting: {target_modules}")


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


def run_training(model_name, yaml_path, ds_path, use_qlora=False):
    """Launch train_custom.py via torchrun."""
    patch_prefetch_factor()

    train_script = os.path.join(COSYVOICE_ROOT, "cosyvoice", "bin", "train_custom.py")
    train_engine = "torch_ddp" if use_qlora else "deepspeed"

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

    if train_engine == "deepspeed":
        cmd.extend(["--deepspeed_config", ds_path])
        cmd.extend(["--deepspeed.save_states", "model_only"])

    if use_qlora:
        cmd.extend(["--use_amp"])

    qwen_path = os.path.join(PRETRAINED_MODEL_DIR, "CosyVoice-BlankEN")
    if not os.path.isdir(qwen_path):
        qwen_path = PRETRAINED_MODEL_DIR
    cmd.extend(["--qwen_pretrain_path", qwen_path])
    cmd.extend(["--onnx_path", PRETRAINED_MODEL_DIR])

    logging.info(f"Running: {' '.join(cmd)}")

    # Environment
    env = os.environ.copy()
    matcha_path = os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{COSYVOICE_ROOT}:{matcha_path}:{existing}"

    # WandB
    if USE_WANDB and WANDB_API_KEY:
        env["WANDB_API_KEY"] = WANDB_API_KEY
        env["WANDB_PROJECT"] = WANDB_PROJECT
        env["WANDB_RUN_NAME"] = WANDB_RUN_NAME
        env["WANDB_MODE"] = "online"

    # Training config (read by train_custom.py via os.environ)
    env["MAX_STEPS"] = str(MAX_STEPS)
    env["LOG_INTERVAL"] = str(LOG_INTERVAL)
    env["SAVE_PER_STEP"] = str(SAVE_PER_STEP)
    env["KEEP_CHECKPOINTS"] = str(KEEP_CHECKPOINTS)
    if EARLY_STOPPING:
        env["EARLY_STOPPING_PATIENCE"] = str(EARLY_STOPPING_PATIENCE)

    result = subprocess.run(cmd, cwd=COSYVOICE_ROOT, env=env)
    if result.returncode != 0:
        logging.error(f"Training failed for model={model_name} with return code {result.returncode}")
        sys.exit(result.returncode)
    logging.info(f"Training completed for model={model_name}")
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


def run_training(model_name, yaml_path, ds_path, use_qlora=False):
    """Launch train_custom.py via torchrun."""
    patch_prefetch_factor()

    train_script = os.path.join(COSYVOICE_ROOT, "cosyvoice", "bin", "train_custom.py")
    train_engine = "torch_ddp" if use_qlora else "deepspeed"

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

    # DeepSpeed-specific args (only when not using QLoRA)
    if train_engine == "deepspeed":
        cmd.extend(["--deepspeed_config", ds_path])
        cmd.extend(["--deepspeed.save_states", "model_only"])

    # torch_ddp with AMP for QLoRA
    if use_qlora:
        cmd.extend(["--use_amp"])

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

    # Pass early stopping + max steps + checkpoint config
    if EARLY_STOPPING:
        env["EARLY_STOPPING_PATIENCE"] = str(EARLY_STOPPING_PATIENCE)
    env["MAX_STEPS"] = str(MAX_STEPS)
    env["KEEP_CHECKPOINTS"] = str(KEEP_CHECKPOINTS)

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

    # 3. If QLoRA, patch YAML to use llm_lora and ensure the file exists
    if USE_QLORA and TRAIN_LLM:
        ensure_llm_lora_exists()
        yaml_path = patch_yaml_for_qlora(yaml_path, tmp_dir)
        logging.info(f"QLoRA enabled with mode={QLORA_MODE}, r={LORA_R}, alpha={LORA_ALPHA}")

    # 4. Run LLM training (required)
    if TRAIN_LLM:
        logging.info("=" * 50)
        logging.info("Stage 1: Training LLM")
        logging.info("=" * 50)
        run_training("llm", yaml_path, ds_path, use_qlora=USE_QLORA)

    # 5. Run Flow training (optional)
    if TRAIN_FLOW:
        logging.info("=" * 50)
        logging.info("Stage 2: Training Flow")
        logging.info("=" * 50)
        # Flow uses original YAML (no QLoRA) with DeepSpeed
        flow_yaml = patch_yaml_train_conf(base_yaml, tmp_dir)
        run_training("flow", flow_yaml, ds_path, use_qlora=False)

    # WandB runs inside the torchrun subprocess and handles its own cleanup

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info("All training stages complete!")


if __name__ == '__main__':
    main()
