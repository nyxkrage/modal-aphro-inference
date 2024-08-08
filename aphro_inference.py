import modal
from common import APHRO_IMAGE, GPU_CLASS, IDLE_TIMEOUT, MODELS_VOLUME, MODELS_DIR, MODEL_NAME, MAX_CONTENT, NGPU, TOKEN

try:
    volume = modal.Volume.lookup(MODELS_VOLUME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model.py")

app = modal.App("aphro-openai-server")

@app.function(
    image=APHRO_IMAGE,
    gpu=GPU_CLASS,
    container_idle_timeout=IDLE_TIMEOUT,
    timeout=24 * 60 * 60,
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
        tensor_parallel_size=NGPU,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_CONTENT,
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
