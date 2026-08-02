"""Local-only FastAPI service for the AI4Mars Research Console."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .repository import RunNotFoundError, RunRepository, UnsafeArtifactPathError
from .schema import RunStatus
from .telemetry import SystemTelemetrySampler


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def create_app(repo_root: Path | None = None, runs_root: Path | None = None) -> FastAPI:
    root = Path(repo_root) if repo_root is not None else _repository_root()
    repository = RunRepository(root, runs_root)
    telemetry = SystemTelemetrySampler()
    app = FastAPI(title="AI4Mars Research Console", version="0.1.0")
    app.state.repository = repository
    app.state.telemetry = telemetry

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "runs_root": repository.runs_root.as_posix(),
            "system": telemetry.collect().model_dump(mode="json", exclude_none=True),
        }

    @app.get("/api/overview")
    def overview() -> dict:
        return repository.overview()

    @app.get("/api/provenance")
    def provenance() -> dict:
        return repository.provenance()

    @app.get("/api/runs")
    def list_runs(
        status: RunStatus | None = None,
        protocol_valid: bool | None = None,
        query: str | None = None,
        manifest_hash: str | None = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict:
        return repository.list_runs(
            status=status,
            protocol_valid=protocol_valid,
            query=query,
            manifest_hash=manifest_hash,
            offset=offset,
            limit=limit,
        )

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        try:
            return repository.detail(run_id)
        except (RunNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Run not found or malformed.") from error

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, after: Annotated[int, Query(ge=0)] = 0) -> dict:
        try:
            return repository.events(run_id, after=after)
        except (RunNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Run not found or malformed.") from error

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run_events(run_id: str, after: Annotated[int, Query(ge=0)] = 0) -> StreamingResponse:
        try:
            repository.events(run_id, after=after)
        except (RunNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Run not found or malformed.") from error

        async def event_stream():
            cursor = after
            while True:
                payload = repository.events(run_id, after=cursor)
                cursor = payload["next"]
                if payload["events"]:
                    yield f"event: run\ndata: {json.dumps(payload)}\n\n"
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/compare")
    def compare(run_id: Annotated[list[str], Query(min_length=2)]) -> dict:
        try:
            return repository.compare(run_id)
        except (RunNotFoundError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/runs/{run_id}/samples")
    def samples(
        run_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        sort_by: str = "image_iou",
        split: str | None = None,
        big_rock_false_negative: bool = False,
        big_rock_to_soil: bool = False,
    ) -> dict:
        try:
            return repository.samples(
                run_id,
                offset=offset,
                limit=limit,
                sort_by=sort_by,
                split=split,
                big_rock_false_negative=big_rock_false_negative,
                big_rock_to_soil=big_rock_to_soil,
            )
        except (RunNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Run not found or malformed.") from error

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str) -> FileResponse:
        try:
            return FileResponse(repository.artifact_path(run_id, artifact_path))
        except (RunNotFoundError, UnsafeArtifactPathError) as error:
            raise HTTPException(status_code=404, detail="Artifact not found.") from error

    frontend_dir = root / "web" / "dist"
    assets_dir = frontend_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    if frontend_dir.is_dir() and (frontend_dir / "index.html").is_file():

        @app.get("/{frontend_path:path}", include_in_schema=False)
        def frontend(frontend_path: str):
            candidate = (frontend_dir / frontend_path).resolve()
            if frontend_path and candidate.is_file() and frontend_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(frontend_dir / "index.html")

    return app


app = create_app()