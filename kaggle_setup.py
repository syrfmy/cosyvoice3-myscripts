"""
📦 CosyVoice Kaggle Environment Setup
Run this as the FIRST cell in your Kaggle notebook:
    !python kaggle_setup.py
"""
import os
import subprocess
import sys


def run(cmd, cwd=None):
    """Run a shell command, exit on failure."""
    print(f"\n{'='*60}\n▶ {cmd}\n{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    working_dir = "/kaggle/working"
    cosyvoice_dir = os.path.join(working_dir, "CosyVoice")

    # --- Step 1: Clean up any previous installs ---
    os.chdir(working_dir)
    if os.path.exists(cosyvoice_dir):
        print("🧹 Removing existing CosyVoice directory...")
        run(f"rm -rf {cosyvoice_dir}")

    # --- Step 2: Install uv (fast package manager) ---
    run("pip install uv -q")

    # --- Step 3: Clone repo with submodules ---
    run("git clone --recurse-submodules https://github.com/FunAudioLLM/CosyVoice.git",
        cwd=working_dir)
    os.chdir(cosyvoice_dir)

    # --- Step 4: Python 3.12 compatibility ---
    run("uv pip install --system setuptools wheel")

    # --- Step 5: Install training dependencies ---
    packages = [
        "conformer==0.3.2",
        "datasets==2.21.0",
        "deepspeed==0.15.1",
        "diffusers==0.29.0",
        "hydra-core==1.3.2",
        "HyperPyYAML==1.2.3",
        "lightning==2.2.4",
        "numpy>=1.26,<2",
        "omegaconf==2.3.0",
        "onnxruntime-gpu==1.18.0",
        "openai-whisper",
        "pyarrow==18.1.0",
        "pyworld==0.3.4",
        "soundfile==0.12.1",
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.51.3",
        "wget==3.2",
        "x-transformers==2.11.24",
        "peft",
        "bitsandbytes",
        "wandb",
    ]
    pkg_str = " ".join(f'"{p}"' for p in packages)
    run(f"uv pip install --system {pkg_str}")

    # --- Step 6: Install Matcha-TTS (skip piper-phonemize) ---
    matcha_dir = os.path.join(cosyvoice_dir, "third_party", "Matcha-TTS")
    run("uv pip install --system --no-build-isolation --no-deps -e .", cwd=matcha_dir)

    print("\n" + "=" * 60)
    print("✅ Setup complete! You can now run:")
    print("   !python kaggle_hf_prep.py   # data preparation")
    print("   !python kaggle_train.py     # training")
    print("=" * 60)


if __name__ == "__main__":
    main()
