import os, logging, shutil, torch, torchaudio
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ==========================================
# 🎛️ INFERENCE CONFIGURATION
# ==========================================

# --- Model Paths ---
PRETRAINED_MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"   # Original pretrained model (for flow.pt, hift.pt, configs)
FINETUNED_LLM_DIR = "/kaggle/working/output/llm/best_model"      # DeepSpeed checkpoint directory from training

# --- Which components were fine-tuned? ---
FINETUNED_LLM = True         # Did you fine-tune the LLM? (matches TRAIN_LLM in kaggle_train.py)
FINETUNED_FLOW = False        # Did you fine-tune the Flow? (matches TRAIN_FLOW in kaggle_train.py)
FINETUNED_FLOW_DIR = "/kaggle/working/output/flow/best_model"     # DeepSpeed checkpoint for flow (if applicable)

# --- Inference Mode ---
# Options: "zero_shot", "cross_lingual", "instruct"
MODE = "zero_shot"

# --- Input (adjust based on MODE) ---
TEXT = "Hello, this is a test sentence."              # Text to synthesize
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>This is a reference prompt."  # Prompt text
PROMPT_AUDIO = "./asset/zero_shot_prompt.wav"          # Path to reference audio
INSTRUCT_TEXT = ""                                      # Instruction text (for instruct mode)

# --- Output ---
OUTPUT_FILE = "./output.wav"

# --- Generation ---
STREAM = False

# ==========================================
# 🔧 MODEL PREPARATION
# ==========================================

def convert_deepspeed_checkpoint(ds_checkpoint_dir, output_pt_path):
    """Convert a DeepSpeed checkpoint directory to a flat state_dict .pt file.
    
    DeepSpeed saves checkpoints as directories containing:
      - mp_rank_00_model_states.pt  (model weights + client_state)
    
    CosyVoice's model.load() expects a flat state_dict .pt file.
    This function extracts the model weights and saves them in the expected format.
    """
    ds_model_file = os.path.join(ds_checkpoint_dir, 'mp_rank_00_model_states.pt')
    
    if not os.path.exists(ds_model_file):
        raise FileNotFoundError(
            f"DeepSpeed model states not found at: {ds_model_file}\n"
            f"Contents of {ds_checkpoint_dir}: {os.listdir(ds_checkpoint_dir) if os.path.isdir(ds_checkpoint_dir) else 'NOT A DIR'}"
        )
    
    logging.info(f"Loading DeepSpeed checkpoint: {ds_model_file}")
    checkpoint = torch.load(ds_model_file, map_location='cpu', weights_only=False)
    
    # DeepSpeed stores model weights under 'module' key
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
        logging.info(f"Extracted 'module' state_dict with {len(state_dict)} keys")
    else:
        # Fallback: assume the checkpoint IS the state dict
        state_dict = checkpoint
        logging.info(f"Using checkpoint directly as state_dict with {len(state_dict)} keys")
    
    # Save as flat state_dict (what CosyVoice expects)
    torch.save(state_dict, output_pt_path)
    logging.info(f"Converted checkpoint saved to: {output_pt_path}")
    return output_pt_path


def prepare_model_dir(pretrained_dir, finetuned_llm_dir=None, finetuned_flow_dir=None):
    """Prepare a model directory that CosyVoice's AutoModel can load.
    
    Strategy:
    1. Copy the pretrained model directory to a working location
    2. Replace llm.pt and/or flow.pt with converted fine-tuned weights
    """
    # Work in a temporary inference directory
    inference_dir = "/kaggle/working/inference_model"
    
    if os.path.exists(inference_dir):
        shutil.rmtree(inference_dir)
    
    logging.info(f"Copying pretrained model from {pretrained_dir} to {inference_dir}")
    shutil.copytree(pretrained_dir, inference_dir)
    
    # Replace LLM weights if fine-tuned
    if finetuned_llm_dir and os.path.isdir(finetuned_llm_dir):
        logging.info("Converting fine-tuned LLM checkpoint...")
        llm_pt = os.path.join(inference_dir, 'llm.pt')
        convert_deepspeed_checkpoint(finetuned_llm_dir, llm_pt)
    
    # Replace Flow weights if fine-tuned
    if finetuned_flow_dir and os.path.isdir(finetuned_flow_dir):
        logging.info("Converting fine-tuned Flow checkpoint...")
        flow_pt = os.path.join(inference_dir, 'flow.pt')
        convert_deepspeed_checkpoint(finetuned_flow_dir, flow_pt)
    
    logging.info(f"Inference model directory ready: {inference_dir}")
    return inference_dir


# ==========================================
# 🚀 MAIN
# ==========================================

def main():
    from cosyvoice.cli.cosyvoice import AutoModel

    # Step 1: Prepare model dir (convert DeepSpeed checkpoints to flat .pt files)
    model_dir = prepare_model_dir(
        pretrained_dir=PRETRAINED_MODEL_DIR,
        finetuned_llm_dir=FINETUNED_LLM_DIR if FINETUNED_LLM else None,
        finetuned_flow_dir=FINETUNED_FLOW_DIR if FINETUNED_FLOW else None,
    )

    # Step 2: Load model using CosyVoice's standard AutoModel
    logging.info(f"Loading model from: {model_dir}")
    cosyvoice = AutoModel(model_dir=model_dir)

    # Step 3: Run inference
    logging.info(f"Running inference in mode: {MODE}")

    if MODE == "zero_shot":
        results = cosyvoice.inference_zero_shot(
            TEXT, PROMPT_TEXT, PROMPT_AUDIO, stream=STREAM
        )
    elif MODE == "cross_lingual":
        results = cosyvoice.inference_cross_lingual(
            TEXT, PROMPT_AUDIO, stream=STREAM
        )
    elif MODE == "instruct":
        results = cosyvoice.inference_instruct2(
            TEXT, INSTRUCT_TEXT, PROMPT_AUDIO, stream=STREAM
        )
    else:
        raise ValueError(f"Unknown MODE: {MODE}. Use 'zero_shot', 'cross_lingual', or 'instruct'.")

    for i, chunk in enumerate(results):
        out_path = OUTPUT_FILE if i == 0 else OUTPUT_FILE.replace('.wav', f'_{i}.wav')
        torchaudio.save(out_path, chunk['tts_speech'], cosyvoice.sample_rate)
        logging.info(f"Saved: {out_path}")

    logging.info("Inference complete!")


if __name__ == '__main__':
    main()
