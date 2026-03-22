import os
os.environ["HF_AUDIO_DECODER"] = "soundfile"  # Use soundfile instead of torchcodec (incompatible with torch 2.3.1)

import logging, shutil, tempfile
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ==========================================
# 🎛️ INFERENCE CONFIGURATION
# ==========================================

# --- Pretrained Model (base CosyVoice3 with configs, flow.pt, hift.pt, etc.) ---
PRETRAINED_MODEL_DIR = "/kaggle/input/datasets/rustedpipe/cosyvoice3-pretrained/Fun-CosyVoice3-0.5B"

# --- Fine-Tuned Checkpoint (from HuggingFace or local) ---
# Set to a HuggingFace repo ID (e.g. "your_username/cosyvoice3-finetuned-llm") or a local directory path.
# If set to None, inference uses the pretrained model without any fine-tuning.
HF_FINETUNED_LLM_REPO = None       # e.g. "your_username/cosyvoice3-finetuned-llm"
HF_FINETUNED_FLOW_REPO = None      # e.g. "your_username/cosyvoice3-finetuned-flow" (if you trained flow)

# --- HuggingFace Test Dataset ---
HF_DATASET = "your_username/your_dataset"    # HuggingFace dataset ID
TEST_SPLIT = "test"                           # Dataset split to run inference on

# --- Dataset Column Names (must match kaggle_hf_prep.py) ---
COL_SOURCE_AUDIO = "source_audio"       # Column with source/prompt audio
COL_SOURCE_ATTR = "source_attribute"    # Column with source attributes (dict with 'teks', etc.)
COL_INSTRUCTION = "instruction"         # Column with editing instruction

# --- Inference Mode ---
# Options: "instruct" (uses instruction + source audio), "zero_shot" (uses text + prompt audio)
MODE = "instruct"

# --- Output ---
OUTPUT_DIR = "/kaggle/working/inference_output"
MAX_SAMPLES = None  # Set to an integer (e.g. 10) to limit inference to N samples, None = all

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
        state_dict = checkpoint
        logging.info(f"Using checkpoint directly as state_dict with {len(state_dict)} keys")

    torch.save(state_dict, output_pt_path)
    logging.info(f"Converted checkpoint saved to: {output_pt_path}")
    return output_pt_path


def download_hf_checkpoint(repo_id):
    """Download a HuggingFace model repo and return the local directory path."""
    from huggingface_hub import snapshot_download
    logging.info(f"Downloading fine-tuned checkpoint from HuggingFace: {repo_id}")
    local_dir = snapshot_download(repo_id)
    logging.info(f"Downloaded to: {local_dir}")
    return local_dir


def prepare_model_dir():
    """Prepare a model directory that CosyVoice's AutoModel can load.

    Strategy:
    1. Copy the pretrained model directory to a working location
    2. Replace llm.pt and/or flow.pt with converted fine-tuned weights (if available)
    """
    inference_model_dir = "/kaggle/working/inference_model"

    if os.path.exists(inference_model_dir):
        shutil.rmtree(inference_model_dir)

    logging.info(f"Copying pretrained model from {PRETRAINED_MODEL_DIR} to {inference_model_dir}")
    shutil.copytree(PRETRAINED_MODEL_DIR, inference_model_dir)

    # Replace LLM weights if fine-tuned
    if HF_FINETUNED_LLM_REPO:
        finetuned_dir = download_hf_checkpoint(HF_FINETUNED_LLM_REPO)

        llm_pt = os.path.join(inference_model_dir, 'llm.pt')
        ds_model_file = os.path.join(finetuned_dir, 'mp_rank_00_model_states.pt')

        if os.path.exists(ds_model_file):
            # DeepSpeed checkpoint directory → convert
            logging.info("Converting fine-tuned LLM DeepSpeed checkpoint...")
            convert_deepspeed_checkpoint(finetuned_dir, llm_pt)
        elif os.path.exists(os.path.join(finetuned_dir, 'llm.pt')):
            # Already a flat .pt file → copy directly
            logging.info("Copying fine-tuned llm.pt directly...")
            shutil.copy2(os.path.join(finetuned_dir, 'llm.pt'), llm_pt)
        else:
            logging.warning(f"Could not find model weights in {finetuned_dir}. Using pretrained LLM.")

    # Replace Flow weights if fine-tuned
    if HF_FINETUNED_FLOW_REPO:
        finetuned_dir = download_hf_checkpoint(HF_FINETUNED_FLOW_REPO)

        flow_pt = os.path.join(inference_model_dir, 'flow.pt')
        ds_model_file = os.path.join(finetuned_dir, 'mp_rank_00_model_states.pt')

        if os.path.exists(ds_model_file):
            logging.info("Converting fine-tuned Flow DeepSpeed checkpoint...")
            convert_deepspeed_checkpoint(finetuned_dir, flow_pt)
        elif os.path.exists(os.path.join(finetuned_dir, 'flow.pt')):
            logging.info("Copying fine-tuned flow.pt directly...")
            shutil.copy2(os.path.join(finetuned_dir, 'flow.pt'), flow_pt)
        else:
            logging.warning(f"Could not find model weights in {finetuned_dir}. Using pretrained Flow.")

    logging.info(f"Inference model directory ready: {inference_model_dir}")
    return inference_model_dir


# ==========================================
# � AUDIO HELPERS
# ==========================================

def save_prompt_audio_to_temp(item, col_name):
    """Save a HuggingFace audio column to a temporary WAV file (required by CosyVoice API)."""
    audio_field = item[col_name]
    arr = np.array(audio_field['array'], dtype=np.float32)
    sr = audio_field['sampling_rate']
    tensor = torch.from_numpy(arr).unsqueeze(0)

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    torchaudio.save(tmp.name, tensor, sr)
    return tmp.name


# ==========================================
# 🚀 MAIN
# ==========================================

def main():
    from cosyvoice.cli.cosyvoice import AutoModel
    from datasets import load_dataset, Audio

    # --- Step 1: Prepare model ---
    model_dir = prepare_model_dir()

    logging.info(f"Loading CosyVoice model from: {model_dir}")
    cosyvoice = AutoModel(model_dir=model_dir)

    # --- Step 2: Load test dataset ---
    logging.info(f"Loading HuggingFace dataset: {HF_DATASET} (split={TEST_SPLIT})")
    ds = load_dataset(HF_DATASET, split=TEST_SPLIT)
    ds = ds.cast_column(COL_SOURCE_AUDIO, Audio(sampling_rate=16000))

    total = len(ds)
    if MAX_SAMPLES is not None:
        total = min(total, MAX_SAMPLES)
    logging.info(f"Running inference on {total} samples (mode={MODE})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 3: Iterate and run inference ---
    for idx in tqdm(range(total), desc="Inference"):
        item = ds[idx]

        # Get text from source_attribute
        src_attr = item.get(COL_SOURCE_ATTR, {})
        text = src_attr.get('teks', '') if isinstance(src_attr, dict) else ''
        instruction = item.get(COL_INSTRUCTION, '')

        # Save prompt audio to a temporary WAV file (CosyVoice API requires a file path)
        prompt_wav_path = save_prompt_audio_to_temp(item, COL_SOURCE_AUDIO)

        try:
            if MODE == "instruct":
                # instruct mode: instruction tells the model what to change about the voice
                results = cosyvoice.inference_instruct2(
                    text, instruction, prompt_wav_path, stream=False
                )
            elif MODE == "zero_shot":
                # zero_shot mode: clone the voice from prompt audio
                prompt_text = text  # Use the transcript as prompt text
                results = cosyvoice.inference_zero_shot(
                    text, prompt_text, prompt_wav_path, stream=False
                )
            elif MODE == "cross_lingual":
                results = cosyvoice.inference_cross_lingual(
                    text, prompt_wav_path, stream=False
                )
            else:
                raise ValueError(f"Unknown MODE: {MODE}")

            # Save all output chunks
            for chunk_idx, chunk in enumerate(results):
                suffix = f"_{chunk_idx}" if chunk_idx > 0 else ""
                out_path = os.path.join(OUTPUT_DIR, f"sample_{idx:04d}{suffix}.wav")
                torchaudio.save(out_path, chunk['tts_speech'], cosyvoice.sample_rate)

            logging.info(f"[{idx+1}/{total}] Saved sample_{idx:04d}.wav | text='{text[:50]}...'")

        except Exception as e:
            logging.error(f"[{idx+1}/{total}] Failed on sample {idx}: {e}")

        finally:
            # Clean up temp file
            if os.path.exists(prompt_wav_path):
                os.unlink(prompt_wav_path)

    logging.info(f"\nInference complete! {total} samples saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
