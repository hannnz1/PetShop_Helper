"""FastAPI application factory and owned runtime resources."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_core.language_models import BaseChatModel

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.config import Settings, get_settings
from app.core.llm import get_chat_model
from app.core.memory import SessionStore

_INDEX = Path(__file__).parent / "static" / "index.html"


def create_app(settings: Settings | None = None, model: BaseChatModel | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            config = settings if settings is not None else get_settings()
        except Exception:
            raise RuntimeError(
                "CHAT_MODEL, CHAT_BASE_URL and CHAT_API_KEY must be configured"
            ) from None

        owned_model = model is None
        try:
            shared_model = model if model is not None else get_chat_model(
                streaming=True, settings=config
            )
        except Exception:
            raise RuntimeError("Unable to initialize chat model") from None

        application.state.settings = config
        application.state.model = shared_model
        application.state.store = SessionStore()
        application.state.active_sessions = set()
        application.state.active_session_owners = {}
        try:
            yield
        finally:
            if owned_model:
                root_async_client = getattr(shared_model, "root_async_client", None)
                root_client = getattr(shared_model, "root_client", None)
                if root_async_client is not None:
                    await root_async_client.close()
                if root_client is not None:
                    root_client.close()

    application = FastAPI(title="PetShop_Helper", version="0.1.0", lifespan=lifespan)
    application.include_router(chat_router)
    application.include_router(extract_router)

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_INDEX)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
