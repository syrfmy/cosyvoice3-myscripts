import os, logging, torchaudio
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ==========================================
# 🎛️ INFERENCE CONFIGURATION
# ==========================================

# --- Model ---
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"   # Path to model directory

# --- Inference Mode ---
# Options: "zero_shot", "cross_lingual", "instruct"
MODE = "zero_shot"

# --- Input (adjust based on MODE) ---
TEXT = "Hello, this is a test sentence."              # Text to synthesize
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>This is a reference prompt."  # Prompt text
PROMPT_AUDIO = "./asset/zero_shot_prompt.wav"          # Path to reference audio
INSTRUCT_TEXT = ""                                      # Instruction text (for instruct mode, put it in PROMPT_TEXT)

# --- Output ---
OUTPUT_FILE = "./output.wav"

# --- Generation ---
STREAM = False

# ==========================================
# 🚀 MAIN
# ==========================================

def main():
    from cosyvoice.cli.model import CosyVoice3Model
    from cosyvoice.cli.cosyvoice import AutoModel

    logging.info(f"Loading model from: {MODEL_DIR}")
    cosyvoice = AutoModel(model_dir=MODEL_DIR)

    logging.info(f"Running inference in mode: {MODE}")

    if MODE == "zero_shot":
        results = cosyvoice.inference_zero_shot(
            TEXT, PROMPT_TEXT, PROMPT_AUDIO, stream=STREAM
        )
    elif MODE == "cross_lingual":
        results = cosyvoice.inference_cross_lingual(
            PROMPT_TEXT, PROMPT_AUDIO, stream=STREAM
        )
    elif MODE == "instruct":
        results = cosyvoice.inference_instruct2(
            TEXT, PROMPT_TEXT, PROMPT_AUDIO, stream=STREAM
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
