import modal
from common import MODELS_VOLUME, MODELS_DIR, MODEL_NAME, APHRO_IMAGE, GPU_CLASS, MODELS_VOLUME, MODELS_DIR, MODEL_NAME, NGPU

try:
    volume = modal.Volume.lookup(MODELS_VOLUME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model.py")

app = modal.App(
    image=APHRO_IMAGE
)
@app.function(volumes={MODELS_DIR: volume}, timeout=4 * 60 * 60, gpu=GPU_CLASS)
def tensorize_model():
    from aphrodite.engine.args_tools import EngineArgs
    from aphrodite.modeling.model_loader.tensorizer import TensorizerConfig, tensorize_aphrodite_model

    volume.reload()

    engine_args = EngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        max_model_len=512,
        tensor_parallel_size=NGPU,
        disable_custom_all_reduce=True
    )

    if NGPU > 1:
        tensorizer_config = TensorizerConfig(tensorizer_uri=MODELS_DIR + "/" + MODEL_NAME + "/model-%03d.tensors")
    else:
        tensorizer_config = TensorizerConfig(tensorizer_uri=MODELS_DIR + "/" + MODEL_NAME + "/model.tensors")

    tensorize_aphrodite_model(engine_args, tensorizer_config)

    volume.commit()


@app.local_entrypoint()
def main():
    tensorize_model.remote()