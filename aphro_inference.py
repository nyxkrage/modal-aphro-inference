import modal

aphro_image = modal.Image.from_registry(tag="ubuntu:jammy", add_python="3.11").pip_install(
    "https://pid1.sh/ai/aphrodite_engine-0.5.4.dev0-cp311-cp311-linux_x86_64.whl",
    "tensorizer>=2.9.0",
)

MODELS_DIR = "/models"
MODEL_NAME = "anthracite-org/magnum-32b-v2"
MODEL_REVISION = "9db035c0017446149f02b742a8f3c2fc896588bf"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

try:
    volume = modal.Volume.lookup("models", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_llama.py")



app = modal.App("aphro-openai-server")

N_GPU = 1
TOKEN = "super-secret-token"  # auth token. for production use, replace with a modal.Secret


@app.function(
    image=aphro_image,
    gpu=modal.gpu.A100(count=N_GPU, size="80GB"),
    container_idle_timeout=1 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve():
    import fastapi
    import aphrodite.endpoints.openai.api_server as api_server
    from aphrodite.engine.async_aphrodite import AsyncEngineArgs, AsyncAphrodite as AsyncLLMEngine
    from aphrodite.endpoints.openai.serving_chat import OpenAIServingChat
    from aphrodite.endpoints.openai.serving_completions import (
        OpenAIServingCompletion,
    )

    volume.reload()

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with Aphrodite-Engine",
        version="1.0.0",
    )

    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    router.include_router(api_server.router)
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        enforce_eager=True,
        load_format="tensorizer",
        model_loader_extra_config={"tensorizer_uri": MODELS_DIR + "/" + MODEL_NAME + "/model.tensors"}
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args
    )

    api_server.openai_serving_chat = OpenAIServingChat(
        engine,
        served_model_names=[MODEL_NAME],
        response_role="assistant",
    )
    api_server.openai_serving_completion = OpenAIServingCompletion(
        engine,
        served_model_names=[MODEL_NAME],
    )

    return web_app
