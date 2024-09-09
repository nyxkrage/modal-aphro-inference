MODEL_NAME="anthracite-org/magnum-32b-v2" # HF Model ID
MODEL_REVISION="9db035c0017446149f02b742a8f3c2fc896588bf" # HF Model Revision
IDLE_TIMEOUT=60 # How long the serverless function should stay warm
NGPU=1 # Number of GPUs to use
GPU="A100:80GB" # Which GPUs to use, for A100 specify 40 or 80GB version like A100:80GB
MODELS_VOLUME="models" # Modal volume to store models on
TOKEN="super-secret-token" # Token to be used to authenticate for users 
MAX_CONTENT=8192 # Max content per request
ENSURE_SAFETENSORS=True # Avoid downloading pytorch .bin files or other non-safetensor weights

import modal
APHRO_IMAGE = modal.Image.from_registry(tag="ubuntu:jammy", add_python="3.11").pip_install(
    "https://pid1.sh/ai/aphrodite_engine-0.5.4.dev0-cp311-cp311-linux_x86_64.whl",
    "tensorizer>=2.9.0",
)
MODELS_DIR="/models"
if ":" in GPU:
    GPU_CLASS=getattr(modal.gpu, GPU.split(":")[0])(count=NGPU, size=GPU.split(":")[1])
else:
    GPU_CLASS=getattr(modal.gpu, GPU)(count=NGPU)