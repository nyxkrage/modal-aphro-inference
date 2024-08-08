# ---
# args: ["--force-download"]
# ---
import modal

MODELS_DIR = "/models"
MODEL_NAME = "anthracite-org/magnum-32b-v2"
MODEL_REVISION = "9db035c0017446149f02b742a8f3c2fc896588bf"
MINUTES = 60
HOURS = 60 * MINUTES

volume = modal.Volume.from_name("models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(
    image=image
)

@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, model_revision, force_download=False):
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        revision=model_revision,
        force_download=force_download,
    )

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = MODEL_NAME,
    model_revision: str = MODEL_REVISION,
    force_download: bool = False,
):
    download_model.remote(model_name, model_revision, force_download)