import modal
from common import MODELS_VOLUME, MODELS_DIR, MODEL_NAME, MODEL_REVISION, ENSURE_SAFETENSORS

volume = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
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

@app.function(volumes={MODELS_DIR: volume}, timeout=4 * 60 * 60)
def download_model():
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        MODEL_NAME,
        local_dir=MODELS_DIR + "/" + MODEL_NAME,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ] if ENSURE_SAFETENSORS else [],
        revision=MODEL_REVISION,
    )

    volume.commit()


@app.local_entrypoint()
def main():
    download_model.remote()