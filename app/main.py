"""FastAPI application factory and owned runtime resources."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel

from app.api.chat import router as chat_router
from app.api.agent import router as agent_router
from app.api.extract import router as extract_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
from app.api.admin import router as admin_router
from app.api.rageval import router as rageval_router
from app.api.actions import router as actions_router
from app.api.conversations import router as conversations_router
from app.config import Settings, get_settings
from app.core.jobs import JobRunner
from app.core.llm import get_chat_model
from app.core.intent import ModelIntentClassifier
from app.core.memory import SessionStore
from app.core.summarizer import close_summary_tasks, configure_summary_model, schedule_recovery
from app.kb import milvus_client
from app.graph.build import build_graph
from app.graph.runtime import GraphRuntime
from app.observability.tracing import close_tracing
from app.graph.nodes import close_context_log, open_context_log, validate_startup_budget
from app.tools import registry as tool_registry

_INDEX = Path(__file__).parent / "static" / "index.html"
_STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, model: BaseChatModel | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            config = settings if settings is not None else get_settings()
        except Exception:
            raise RuntimeError(
                "CHAT_MODEL, CHAT_BASE_URL and CHAT_API_KEY must be configured"
            ) from None

        tool_registry.scan_builtin()
        validate_startup_budget(config)
        owned_model = model is None
        try:
            shared_model = model if model is not None else get_chat_model(
                streaming=True, settings=config
            )
        except Exception:
            raise RuntimeError("Unable to initialize chat model") from None

        application.state.settings = config
        context_log = open_context_log(Path(__file__).parent.parent / "log" / "app.log")
        application.state.model = shared_model
        configure_summary_model(shared_model, config)
        application.state.store = SessionStore()
        application.state.active_sessions = set()
        application.state.active_session_owners = {}
        application.state.jobs = JobRunner()
        milvus_client.add_runtime_owner()
        try:
            classifier = ModelIntentClassifier(shared_model, config.structured_output_method)
            async with GraphRuntime(Path(config.graph_checkpoint_path), build_graph,
                                    classifier=classifier, settings=config) as graph:
                application.state.graph = graph
                await schedule_recovery(graph)
                yield
        finally:
            await close_summary_tasks()
            try:
                application.state.jobs.stop_all()
            finally:
                try:
                    milvus_client.release_runtime_owner()
                finally:
                    try:
                        if owned_model:
                            root_async_client = getattr(shared_model, "root_async_client", None)
                            root_client = getattr(shared_model, "root_client", None)
                            if root_async_client is not None:
                                await root_async_client.close()
                            if root_client is not None:
                                root_client.close()
                    finally:
                        close_tracing(config)
                        close_context_log(context_log)

    application = FastAPI(title="PetShop_Helper", version="0.1.0", lifespan=lifespan)
    application.include_router(chat_router)
    application.include_router(agent_router)
    application.include_router(extract_router)
    application.include_router(kb_router)
    application.include_router(jobs_router)
    application.include_router(admin_router)
    application.include_router(rageval_router)
    application.include_router(actions_router)
    application.include_router(conversations_router)
    application.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_INDEX)

    @application.get("/kb", include_in_schema=False)
    def knowledge_page() -> FileResponse:
        return FileResponse(_STATIC / "kb.html")

    @application.get("/admin", include_in_schema=False)
    def admin_page() -> FileResponse:
        return FileResponse(_STATIC / "admin.html")

    @application.get("/rag-eval", include_in_schema=False)
    def rag_eval_page() -> FileResponse:
        return FileResponse(_STATIC / "rageval.html")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
