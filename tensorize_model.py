import modal


MODELS_DIR = "/models"
MODEL_NAME = "anthracite-org/magnum-32b-v2"
MODEL_REVISION = "9db035c0017446149f02b742a8f3c2fc896588bf"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

volume = modal.Volume.from_name("models", create_if_missing=True)

aphro_image = modal.Image.from_registry(tag="ubuntu:jammy", add_python="3.11").pip_install(
    "https://pid1.sh/ai/aphrodite_engine-0.5.4.dev0-cp311-cp311-linux_x86_64.whl",
    "tensorizer>=2.9.0"
)
app = modal.App(
    image=aphro_image
)
@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS, gpu=modal.gpu.A100(count=1, size="80GB"))
def tensorize_model(model_name):
    from aphrodite.engine.args_tools import EngineArgs
    from aphrodite.modeling.model_loader.tensorizer import TensorizerConfig, tensorize_aphrodite_model

    volume.reload()

    engine_args = EngineArgs(
        model=MODELS_DIR + "/" + model_name,
    )

    tensorizer_config = TensorizerConfig(tensorizer_uri=MODELS_DIR + "/" + MODEL_NAME + "/model.tensors")

    tensorize_aphrodite_model(engine_args, tensorizer_config)

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = MODEL_NAME,
):
    tensorize_model.remote(model_name)