"""FastAPI application factory for the ICBT dashboard."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


class _SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(_clean_for_json(content))


def _clean_for_json(o):
    if isinstance(o, float):
        if math.isinf(o) or math.isnan(o):
            return None
        return o
    if isinstance(o, dict):
        return {k: _clean_for_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_for_json(v) for v in o]
    return o


from .. import __version__
from ..analytics import result_summary_text, write_run_artifacts
from ..data import load_csv, validate_ohlcv
from ..engine import BacktestEngine, EngineConfig, FillModel
from ..engine.persistence import (
    RunRecord,
    append_record,
    ensure_runs_dir,
    file_hash,
    load_records,
)
from ..strategy.nfi_x8 import NostalgiaForInfinityX8
from ..strategy.sma_cross import SmaCrossStrategy

STRATEGIES = {
    "SmaCross": SmaCrossStrategy,
    "NFI-X8": NostalgiaForInfinityX8,
}


def create_app(runs_dir: Path = Path("runs"), datasets_dir: Path = Path("data")) -> FastAPI:
    runs_dir = Path(runs_dir)
    datasets_dir = Path(datasets_dir)
    ensure_runs_dir(runs_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="ICBT Dashboard", version=__version__, default_response_class=_SafeJSONResponse
    )
    tpl_dir = Path(__file__).resolve().parent.parent / "ui" / "templates"
    static_dir = Path(__file__).resolve().parent.parent / "ui" / "static"
    templates = Jinja2Templates(directory=str(tpl_dir))
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _context(request: Request, **extra: Any) -> dict[str, Any]:
        return {"request": request, "version": __version__, **extra}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        datasets = _list_datasets(datasets_dir)
        records = load_records(runs_dir)[::-1][:20]
        ctx = _context(request, datasets=datasets, runs=records, strategies=list(STRATEGIES.keys()))
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/api/datasets")
    async def api_datasets():
        return {"datasets": _list_datasets(datasets_dir)}

    @app.get("/api/dataset/{name}")
    async def api_dataset_info(name: str):
        path = _safe_dataset_path(datasets_dir, name)
        data = load_csv(path)
        vr = validate_ohlcv(data)
        return {
            "metadata": {
                "exchange": data.exchange,
                "symbol": data.symbol,
                "timeframe": data.timeframe,
                "rows": len(data),
                "start": data.metadata.start_utc.isoformat() if data.metadata.start_utc else None,
                "end": data.metadata.end_utc.isoformat() if data.metadata.end_utc else None,
                "sha256": data.metadata.source_sha256,
            },
            "validation": vr.to_dict(),
        }

    @app.post("/api/upload")
    async def api_upload(file: UploadFile = File(...)):
        if not file.filename:
            raise HTTPException(400, "no filename")
        safe = Path(file.filename).name  # basename only
        dest = datasets_dir / safe
        with open(dest, "wb") as f:
            f.write(await file.read())
        return {"ok": True, "filename": safe, "path": str(dest)}

    @app.post("/api/validate")
    async def api_validate(payload: dict):
        name = payload.get("file")
        path = _safe_dataset_path(datasets_dir, name)
        data = load_csv(path)
        vr = validate_ohlcv(data)
        return {"report": vr.to_dict(), "metadata": data.metadata.to_dict()}

    @app.post("/api/run")
    async def api_run(payload: dict):
        name = payload.get("dataset")
        if not name:
            raise HTTPException(400, "dataset required")
        path = _safe_dataset_path(datasets_dir, name)
        strat_name = payload.get("strategy", "SmaCross")
        if strat_name not in STRATEGIES:
            raise HTTPException(400, f"unknown strategy {strat_name}")
        strat_cls = STRATEGIES[strat_name]
        strategy = strat_cls()
        btc_path = payload.get("btc_dataset")
        btc_data = None
        if btc_path:
            bp = _safe_dataset_path(datasets_dir, btc_path)
            btc_data = load_csv(
                bp,
                exchange="wallex",
                symbol="BTC/USDT",
                timeframe=payload.get("btc_timeframe") or "5m",
            )
        cfg = EngineConfig(
            initial_cash=float(payload.get("cash", 10_000)),
            stake_pct=float(payload.get("stake_pct", 0.10)),
            stake_amount=(float(payload["stake_amount"]) if payload.get("stake_amount") else None),
            fee_rate=float(payload.get("fee", 0.001)),
            slippage_bps=float(payload.get("slippage_bps", 5)),
            fill_model=FillModel(payload.get("fill_model", "next_open")),
            stoploss=(float(payload["stoploss"]) if payload.get("stoploss") else None),
            max_open_trades=int(payload.get("max_open", 1)),
            start=payload.get("start") or None,
            end=payload.get("end") or None,
            enable_buy_and_hold=not bool(payload.get("no_buyhold", False)),
            fail_on_gaps=bool(payload.get("fail_on_gaps", False)),
        )
        data = load_csv(path)
        engine = BacktestEngine(cfg)
        result = engine.run(data, strategy, btc_data=btc_data)
        paths = write_run_artifacts(result, runs_dir)
        strategy_src = __import__("sys").modules[strat_cls.__module__].__file__
        rec = RunRecord(
            run_id=result.run_id,
            created_at=__import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            strategy_source_hash=file_hash(strategy_src) if strategy_src else "",
            dataset_path=str(path.resolve()),
            dataset_hash=data.metadata.source_sha256 or "",
            exchange=data.exchange,
            symbol=data.symbol,
            timeframe=data.timeframe,
            start_utc=data.metadata.start_utc.isoformat() if data.metadata.start_utc else None,
            end_utc=data.metadata.end_utc.isoformat() if data.metadata.end_utc else None,
            engine_version=__version__,
            config=cfg.to_dict(),
            metrics=result.metrics.to_dict(),
            validation_ok=bool(result.validation.get("ok")),
            warnings=result.warnings,
            errors=result.errors,
            output_dir=str(Path(paths["summary"]).parent),
        )
        append_record(runs_dir, rec)
        eq = []
        if len(result.equity_curve):
            sample = result.equity_curve.iloc[:: max(1, len(result.equity_curve) // 500)]
            eq = [
                {"t": idx.isoformat(), "equity": float(row["equity"]), "close": float(row["close"])}
                for idx, row in sample.iterrows()
            ]
        return {
            "ok": True,
            "run_id": result.run_id,
            "summary": result_summary_text(result),
            "metrics": result.metrics.to_dict(),
            "trades": result.trades[:200],
            "validation": result.validation,
            "warnings": result.warnings,
            "errors": result.errors,
            "equity": eq,
            "paths": paths,
        }

    @app.get("/api/runs")
    async def api_runs():
        return {"runs": load_records(runs_dir)[::-1]}

    @app.get("/api/run/{run_id}")
    async def api_run_detail(run_id: str):
        for r in load_records(runs_dir):
            if r.get("run_id") == run_id:
                # Load artifacts
                out_dir = Path(r.get("output_dir", runs_dir / run_id))
                summary_p = out_dir / "summary.txt"
                summary = summary_p.read_text() if summary_p.exists() else ""
                return {"record": r, "summary": summary}
        raise HTTPException(404, "not found")

    @app.get("/runs/{run_id}/download/{kind}")
    async def run_download(run_id: str, kind: str):
        kind_map = {
            "summary": "summary.txt",
            "trades": "trades.csv",
            "equity": "equity.csv",
            "json": "result.json",
            "config": "config.json",
        }
        if kind not in kind_map:
            raise HTTPException(404, "unknown artifact")
        for r in load_records(runs_dir):
            if r.get("run_id") == run_id:
                out_dir = Path(r["output_dir"])
                p = out_dir / kind_map[kind]
                if not p.exists():
                    raise HTTPException(404, "artifact missing")
                return FileResponse(p, filename=p.name)
        raise HTTPException(404, "not found")

    return app


def _list_datasets(datasets_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not datasets_dir.exists():
        return items
    for p in sorted(datasets_dir.glob("*.csv")):
        try:
            data = load_csv(p)
            vr = validate_ohlcv(data)
            items.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size": p.stat().st_size,
                    "exchange": data.exchange,
                    "symbol": data.symbol,
                    "timeframe": data.timeframe,
                    "rows": len(data),
                    "valid": vr.ok,
                    "warnings": vr.warning_count,
                    "errors": vr.error_count,
                    "gaps": len(vr.gaps),
                    "sha256": data.metadata.source_sha256,
                }
            )
        except Exception as e:
            items.append(
                {"name": p.name, "path": str(p), "size": p.stat().st_size, "error": str(e)}
            )
    return items


def _safe_dataset_path(datasets_dir: Path, name: str) -> Path:
    """Resolve a dataset name to a path inside datasets_dir, rejecting traversal."""
    p = (datasets_dir / Path(name).name).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"dataset {name} not found")
    if datasets_dir.resolve() not in p.parents and p.parent != datasets_dir.resolve():
        raise HTTPException(403, "invalid path")
    return p
