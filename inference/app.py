import os
import sys
import glob
import ctypes
import site
import subprocess
import re

# =============================================================
# 1. Ensure CUDA Runtime Libraries (libcudart.so.12) are linked
# =============================================================
def fix_cuda_libs():
    """Expose bundled NVIDIA CUDA runtime libraries to dlopen for llama-cpp."""
    candidate_dirs = []
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_dir):
            for root, dirs, _ in os.walk(nvidia_dir):
                if "lib" in dirs:
                    candidate_dirs.append(os.path.join(root, "lib"))
    for p in ["/usr/local/cuda/lib64", "/usr/local/cuda/lib"]:
        if os.path.exists(p):
            candidate_dirs.append(p)

    if candidate_dirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(candidate_dirs + ([current] if current else []))

    # Preload libcudart and libcublas using RTLD_GLOBAL so libllama.so finds symbols immediately
    for cdir in candidate_dirs:
        for pattern in ["libcudart.so*", "libcublas.so*", "libcuda.so*"]:
            for so_file in glob.glob(os.path.join(cdir, pattern)):
                try:
                    ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

def ensure_cuda_runtime():
    fix_cuda_libs()
    try:
        ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
        print("libcudart.so.12 successfully loaded.")
    except OSError:
        print("libcudart.so.12 not found. Installing nvidia-cuda-runtime-cu12...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"
        ])
        fix_cuda_libs()

# =============================================================
# 2. Bootstrap prebuilt CUDA-enabled llama-cpp-python for ZeroGPU
# =============================================================
def bootstrap_llama_cpp():
    print("Bootstrapping llama-cpp-python with CUDA for ZeroGPU...")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files("AIencoder/llama-cpp-wheels", repo_type="dataset")

    cuda_candidates = [
        p for p in files 
        if p.endswith(".whl") 
        and ("cuda12.4" in p or "cuda12.6" in p) 
        and "basic" in p 
        and "/0.3.36/" in p
    ]
    cpu_candidates = [
        p for p in files 
        if p.endswith(".whl") 
        and "linux/x86_64/cpu/" in p 
        and "/0.3.36/" in p
    ]

    chosen = cuda_candidates[0] if cuda_candidates else (cpu_candidates[0] if cpu_candidates else None)
    if not chosen:
        raise RuntimeError("No compatible prebuilt llama-cpp-python wheel found.")

    print(f"Selected wheel for ZeroGPU: {chosen}")
    whl_path = hf_hub_download("AIencoder/llama-cpp-wheels", chosen, repo_type="dataset")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", whl_path
    ])
    print("llama-cpp-python package installed.")

# Make sure CUDA runtime is available in dynamic linker before importing llama_cpp
ensure_cuda_runtime()

try:
    from llama_cpp import Llama
    print("llama-cpp-python loaded successfully.")
except (ImportError, RuntimeError, OSError):
    bootstrap_llama_cpp()
    ensure_cuda_runtime()
    from llama_cpp import Llama

import spaces
from typing import List
import gradio as gr

# =============================================================
# 3. ZeroGPU Lazy Model Loader
# =============================================================
MODEL_PATH = "Qwen3.5-0.8B.Q4_K_M.gguf"
llm = None

def get_llm():
    global llm
    if llm is None:
        print(f"Initializing {MODEL_PATH} on ZeroGPU...")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False
            )
            print("Model loaded successfully with GPU offload!")
        except Exception as e:
            print(f"GPU offload warning: {e}. Falling back to CPU execution.")
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=2,
                n_gpu_layers=0,
                verbose=False
            )
            print("Model loaded in CPU mode.")
    return llm

SYSTEM_PROMPT = (
    "You are Sherpa, an AI driving assistant for the CommitCon ride-hailing "
    "repositioning platform in Bengaluru. Provide concise, direct, helpful "
    "recommendations to drivers about where to reposition for high demand, "
    "earnings, and low traffic delays."
)

# =============================================================
# 4. Inference with ZeroGPU Allocation
# =============================================================
@spaces.GPU(duration=30)
def gradio_chat(user_message, history):
    model = get_llm()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                u, b = item
                messages.append({"role": "user", "content": str(u)})
                if b:
                    messages.append({"role": "assistant", "content": str(b)})
            elif isinstance(item, dict):
                messages.append(item)

    messages.append({"role": "user", "content": user_message})

    response = model.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=256
    )
    return response["choices"][0]["message"]["content"]

# =============================================================
# 5. Gradio Interface (Exposing /chat API)
# =============================================================
demo = gr.ChatInterface(
    fn=gradio_chat,
    title="🧭 CommitCon Sherpa Driver Copilot",
    description="Fine-tuned Qwen 0.8B Q4_K_M on ZeroGPU — llama-cpp-python",
    api_name="chat"
)

if __name__ == "__main__":
    demo.launch()
