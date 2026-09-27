"""Separate lightweight topic-classifier HTTP service (port 8110)."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scripts.ch10.inference_lib import load_runtime


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


def create_app(runtime=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or load_runtime(Path(os.environ.get("CH10_MODEL_DIR", "data/ch10/onnx")))
        yield

    app = FastAPI(title="MewHelp Topic Classifier", lifespan=lifespan)

    @app.post("/classify")
    async def classify(request: ClassifyRequest):
        if any(not text.strip() for text in request.texts):
            raise HTTPException(422, "texts must be nonblank")
        return {"results": app.state.runtime.classify(request.texts)}

    return app


app = create_app()
