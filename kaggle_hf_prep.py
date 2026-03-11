import os
os.environ["HF_AUDIO_DECODER"] = "soundfile"  # Use soundfile instead of torchcodec (incompatible with torch 2.3.1)

import json, logging, io
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import whisper
import pandas as pd
from tqdm import tqdm
from datasets import Audio
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ==========================================
# 🎛️ DATA PREPARATION CONFIGURATION
# ==========================================

# --- HuggingFace Dataset ---
HF_DATASET = "your_username/your_dataset"   # HuggingFace dataset ID
SPLITS = ["train", "validation"]              # Dataset splits to process (e.g. ["train"], ["train", "validation", "test"])

# --- Model Paths ---
PRETRAINED_MODEL_DIR = "/kaggle/input/datasets/rustedpipe/cosyvoice3-pretrained/Fun-CosyVoice3-0.5B"  # Kaggle dataset input

# --- Output ---
OUTPUT_BASE_DIR = "/kaggle/working/data"  # Base output dir (each split gets a subdirectory)
NUM_UTTS_PER_PARQUET = 1000             # Number of utterances per Parquet shard

# --- Dataset Column Names ---
# Adjust these if your HF dataset uses different column names:
COL_SOURCE_AUDIO = "source_audio"       # Column with source audio
COL_TARGET_AUDIO = "target_audio"       # Column with target audio
COL_SOURCE_ATTR = "source_attribute"    # Column with source attributes (dict with 'teks', 'emotion', etc.)
COL_TARGET_ATTR = "target_attribute"    # Column with target attributes
COL_INSTRUCTION = "instruction"         # Column with editing instruction

# ==========================================
# 🔧 HELPER FUNCTIONS (ONNX Feature Extraction)
# ==========================================

def extract_embedding(audio_tensor, sample_rate, ort_session):
    """Extract speaker embedding from audio tensor using campplus ONNX model."""
    if sample_rate != 16000:
        audio_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_tensor)
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
    feat = kaldi.fbank(audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(
        None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
    )[0].flatten().tolist()
    return embedding


def extract_speech_tokens(audio_tensor, sample_rate, ort_session):
    """Extract discrete speech tokens from audio tensor using speech tokenizer ONNX model."""
    if sample_rate != 16000:
        audio_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_tensor)
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
    if audio_tensor.shape[1] / 16000 > 30:
        logging.warning('Audio longer than 30s, may produce poor tokens.')
    feat = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)
    speech_token = ort_session.run(None, {
        ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
        ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
    })[0].flatten().tolist()
    return speech_token


def load_audio_from_item(item, col_name):
    """Load audio from a HuggingFace dataset item (after cast_column to Audio)."""
    audio_field = item[col_name]
    # After cast_column(Audio), this is always {'array': np.ndarray, 'sampling_rate': int}
    arr = np.array(audio_field['array'], dtype=np.float32)
    arr = torch.from_numpy(arr).unsqueeze(0)
    sr = audio_field['sampling_rate']
    return arr, sr


def audio_to_wav_bytes(audio_tensor, sample_rate):
    """Convert an audio tensor to WAV bytes for storage in Parquet."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    try:
        torchaudio.save(tmp.name, audio_tensor, sample_rate)
        with open(tmp.name, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


# ==========================================
# 🚀 MAIN
# ==========================================

def main():
    # --- Init ONNX Sessions (shared across splits) ---
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = 1

    emb_session = None
    onnx_spk = os.path.join(PRETRAINED_MODEL_DIR, 'campplus.onnx')
    if os.path.exists(onnx_spk):
        emb_session = onnxruntime.InferenceSession(onnx_spk, sess_options=opt, providers=["CPUExecutionProvider"])
        logging.info(f"Loaded speaker embedding model: {onnx_spk}")

    tok_session = None
    onnx_speech = os.path.join(PRETRAINED_MODEL_DIR, 'speech_tokenizer_v3.onnx')
    if os.path.exists(onnx_speech):
        tok_session = onnxruntime.InferenceSession(onnx_speech, sess_options=opt, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    # --- Process each split ---
    for split in SPLITS:
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing split: {split}")
        logging.info(f"{'='*50}")

        output_dir = os.path.join(OUTPUT_BASE_DIR, split)
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Loading HuggingFace dataset: {HF_DATASET} (split={split})")
        try:
            ds = load_dataset(HF_DATASET, split=split)
            # Force eager audio decoding (fixes AudioDecoder / torchcodec issue)
            ds = ds.cast_column(COL_SOURCE_AUDIO, Audio(sampling_rate=16000))
            ds = ds.cast_column(COL_TARGET_AUDIO, Audio(sampling_rate=16000))
        except (ValueError, KeyError) as e:
            logging.warning(f"Split '{split}' not found or missing columns, skipping. Error: {e}")
            continue

        process_split(ds, split, output_dir, emb_session, tok_session)

    logging.info("\nAll splits processed!")


def process_split(ds, split, output_dir, emb_session, tok_session):
    """Process a single dataset split into Parquet files."""
    parquet_list = []
    total_len = len(ds)
    logging.info(f"Split '{split}': {total_len} utterances")

    for chunk_start in tqdm(range(0, total_len, NUM_UTTS_PER_PARQUET), desc="Processing"):
        chunk_end = min(chunk_start + NUM_UTTS_PER_PARQUET, total_len)
        batch_ds = ds.select(range(chunk_start, chunk_end))
        parquet_file = os.path.join(output_dir, f'parquet_{chunk_start:09d}.tar')

        rows = []

        for j, item in enumerate(batch_ds):
            utt = f"utt_{chunk_start + j:09d}"

            try:
                # Load source and target audio
                src_audio, src_sr = load_audio_from_item(item, COL_SOURCE_AUDIO)
                tgt_audio, tgt_sr = load_audio_from_item(item, COL_TARGET_AUDIO)

                # Get transcript text from source_attribute
                src_attr = item.get(COL_SOURCE_ATTR, {})
                transcript = src_attr.get('teks', '') if isinstance(src_attr, dict) else ''

                # Get editing instruction
                instruction = item.get(COL_INSTRUCTION, '')

                # Extract speaker embedding from source audio
                embedding = extract_embedding(src_audio, src_sr, emb_session) if emb_session else [0.0] * 192

                # Extract speech tokens from target audio (this is what the LLM learns to predict)
                speech_token = extract_speech_tokens(tgt_audio, tgt_sr, tok_session) if tok_session else []

                # Convert target audio to WAV bytes at original sample rate
                audio_data = audio_to_wav_bytes(tgt_audio, tgt_sr)

                rows.append({
                    'utt': utt,
                    'audio_data': audio_data,
                    'text': transcript,            # transcript text (tokenized by data_pipeline)
                    'instruct': instruction,       # editing instruction (e.g. "change emotion to happy")
                    'speech_token': speech_token,
                    'utt_embedding': embedding,
                    'spk_embedding': embedding,    # same as utt_embedding for single-speaker samples
                })

            except Exception as e:
                logging.warning(f"Failed to process {utt}: {e}")
                continue

        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(parquet_file)
            parquet_list.append(parquet_file)
            logging.info(f"Wrote {len(rows)} utterances to {parquet_file}")

    # Write data.list
    list_path = os.path.join(output_dir, 'data.list')
    with open(list_path, 'w') as f:
        for p in parquet_list:
            f.write(p + '\n')

    logging.info(f"Split '{split}': {len(parquet_list)} parquet files written. data.list at: {list_path}")


if __name__ == '__main__':
    main()
