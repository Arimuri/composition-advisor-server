"""FastAPI server exposing composition_advisor as HTTP endpoints.

Endpoints
---------
GET  /             — simple HTML upload form (one file each for melody/chord/bass)
POST /analyze      — multipart upload of one or more .mid files; returns AnalysisResult JSON
POST /critique     — same as /analyze but also calls Claude and returns the natural-language critique
POST /fix          — same as /analyze; runs the fix pipeline; returns a zip of fixed MIDIs + diff report
GET  /healthz      — liveness probe (no work, no auth)

Auth: HTTP Basic, configured via env vars
    COMPOSITION_ADVISOR_USER, COMPOSITION_ADVISOR_PASSWORD
If those env vars are unset, all routes (except /healthz) require no auth —
useful for local development. In production set them via systemd.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import secrets
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

# Memory mitigation A: only one heavy chordify/species request at a time.
# This bounds the resident-memory peak so the systemd MemoryMax limit is
# not hit when multiple students poke the server simultaneously.
HEAVY_LOCK = asyncio.Semaphore(int(os.environ.get("COMPOSITION_ADVISOR_CONCURRENCY", "1")))

# Memory mitigation D: reject obviously oversized uploads before allocating
# memory. Real counterpoint exercises are tiny (a few KB).
MAX_UPLOAD_BYTES = int(os.environ.get("COMPOSITION_ADVISOR_MAX_UPLOAD", str(2 * 1024 * 1024)))

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import music21 as m21

from composition_advisor.analyze.key_detector import detect_key, parse_key
from composition_advisor.analyze.voice_extractor import extract_slices
from composition_advisor.cli import _annotate_slice_degrees
from composition_advisor.critique.config import load_config
from composition_advisor.critique.runner import run_all as run_all_rules
from composition_advisor.critique.species_runner import run_species
from composition_advisor.fix.applier import apply_fixes_to_midi, write_diff_report
from composition_advisor.fix.llm import propose as propose_llm_fixes
from composition_advisor.fix.rule_based import propose as propose_rule_fixes
from composition_advisor.io.midi_loader import load_midi_files
from composition_advisor.io.normalize import normalize_score
from composition_advisor.llm.claude_client import critique as llm_critique
from composition_advisor.model.issue import AnalysisResult
from composition_advisor.tutor.cantus_firmus import PRESETS as CF_PRESETS, get as get_cf
from composition_advisor.tutor.feedback_prompt import critique_species

logger = logging.getLogger(__name__)
app = FastAPI(title="composition-advisor-server", version="0.1.0")

security = HTTPBasic(auto_error=False)


def _basic_auth(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(security)] = None,
) -> str:
    """Validate HTTP Basic credentials against env vars.

    If COMPOSITION_ADVISOR_USER is unset we accept any caller — handy for
    local development. In production systemd should always inject the var.
    """
    user = os.environ.get("COMPOSITION_ADVISOR_USER")
    pw = os.environ.get("COMPOSITION_ADVISOR_PASSWORD")
    if not user or not pw:
        return "anonymous"
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    ok_user = secrets.compare_digest(credentials.username.encode(), user.encode())
    ok_pw = secrets.compare_digest(credentials.password.encode(), pw.encode())
    if not (ok_user and ok_pw):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def _save_uploads_to_tmp(files: list[UploadFile], dest: Path) -> list[Path]:
    """Persist uploaded files to a temp directory and return their paths.

    Per-file size cap is enforced (default 2 MB) to keep memory bounded.
    """
    saved: list[Path] = []
    for upload in files:
        if not upload.filename:
            continue
        # Strip any directory components for safety.
        name = Path(upload.filename).name
        if not name.lower().endswith(".mid") and not name.lower().endswith(".midi"):
            raise HTTPException(
                status_code=400,
                detail=f"Only .mid / .midi files are accepted (got {name!r})",
            )
        out_path = dest / name
        bytes_written = 0
        with out_path.open("wb") as fh:
            while True:
                chunk = upload.file.read(64 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    fh.close()
                    out_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"{name!r} exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
                    )
                fh.write(chunk)
        saved.append(out_path)
    if not saved:
        raise HTTPException(status_code=400, detail="No MIDI files uploaded")
    return saved


def _resolve_key(m21_score: m21.stream.Score, key_arg: str | None) -> m21.key.Key:
    return parse_key(key_arg) if key_arg else detect_key(m21_score)


import math


def _sanitize_for_json(obj):
    """Recursively replace NaN/Infinity floats with None.

    music21 occasionally produces NaN beat positions for empty measures or
    odd time-signature contexts. starlette's JSONResponse delegates to the
    standard json module which raises ValueError on non-finite floats.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _build_result(
    m21_score: m21.stream.Score, detected_key: m21.key.Key, config_path: Path | None
) -> AnalysisResult:
    cfg = load_config(config_path) if config_path else None
    internal = normalize_score(m21_score, key=detected_key)
    slices = extract_slices(internal)
    _annotate_slice_degrees(slices, detected_key)
    issues = run_all_rules(internal, slices, config=cfg)
    return AnalysisResult(metadata=internal.metadata, slices=slices, issues=issues)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(_: str = Depends(_basic_auth)) -> str:
    return INDEX_HTML


@app.post("/analyze")
async def analyze_endpoint(
    files: Annotated[list[UploadFile], File(...)],
    key: Annotated[str | None, Form()] = None,
    _: str = Depends(_basic_auth),
) -> JSONResponse:
    async with HEAVY_LOCK:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            saved = _save_uploads_to_tmp(files, tmp_dir)
            m21_score = load_midi_files([str(p) for p in saved])
            detected_key = _resolve_key(m21_score, key)
            result = _build_result(m21_score, detected_key, None)
    return JSONResponse(content=_sanitize_for_json(result.model_dump()))


@app.post("/critique")
async def critique_endpoint(
    files: Annotated[list[UploadFile], File(...)],
    key: Annotated[str | None, Form()] = None,
    _: str = Depends(_basic_auth),
) -> JSONResponse:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503, detail="ANTHROPIC_API_KEY is not configured on the server"
        )
    async with HEAVY_LOCK:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            saved = _save_uploads_to_tmp(files, tmp_dir)
            m21_score = load_midi_files([str(p) for p in saved])
            detected_key = _resolve_key(m21_score, key)
            result = _build_result(m21_score, detected_key, None)
            critique_text = llm_critique(result)
    return JSONResponse(
        content={
            "critique": critique_text,
            "issue_count": len(result.issues),
            "key": result.metadata.key,
        }
    )


@app.post("/musicxml")
async def musicxml_endpoint(
    files: Annotated[list[UploadFile], File(...)],
    _: str = Depends(_basic_auth),
) -> Response:
    """Convert uploaded MIDI files into a single merged MusicXML.

    Used by the browser-side OSMD renderer to draw the score.
    """
    async with HEAVY_LOCK:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            saved = _save_uploads_to_tmp(files, tmp_dir)
            m21_score = load_midi_files([str(p) for p in saved])
            out = tmp_dir / "score.musicxml"
            m21_score.write("musicxml", fp=out)
            data = out.read_bytes()
    return Response(
        content=data,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": 'inline; filename="score.musicxml"'},
    )


@app.post("/species")
async def species_endpoint(
    counterpoint: Annotated[UploadFile, File(...)],
    cantus_firmus: Annotated[UploadFile | None, File()] = None,
    preset: Annotated[str | None, Form()] = None,
    species_num: Annotated[int, Form()] = 1,
    key: Annotated[str | None, Form()] = None,
    _: str = Depends(_basic_auth),
) -> JSONResponse:
    """Run Species Counterpoint analysis.

    Provide either a `cantus_firmus` upload or a built-in `preset` name.
    The response includes both the AnalysisResult and a base64-free
    MusicXML payload for the OSMD renderer.
    """
    if cantus_firmus is None and not preset:
        raise HTTPException(
            status_code=400,
            detail="Either cantus_firmus upload or preset name is required",
        )
    if cantus_firmus is not None and preset:
        raise HTTPException(
            status_code=400, detail="Provide either cantus_firmus or preset, not both"
        )

    async with HEAVY_LOCK:
        return await _species_impl(counterpoint, cantus_firmus, preset, species_num, key)


async def _species_impl(
    counterpoint: UploadFile,
    cantus_firmus: UploadFile | None,
    preset: str | None,
    species_num: int,
    key: str | None,
) -> JSONResponse:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Save counterpoint upload
        cp_saved = _save_uploads_to_tmp([counterpoint], tmp_dir)[0]

        if preset:
            if preset not in CF_PRESETS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset {preset!r}. Available: {sorted(CF_PRESETS)}",
                )
            cf_part = get_cf(preset).to_part(part_name="cantus_firmus")
            m21_score = m21.stream.Score()
            m21_score.insert(0, cf_part)
            cp_score = load_midi_files([str(cp_saved)])
            for p in cp_score.parts:
                p.partName = "counterpoint"
                m21_score.insert(0, p)
        else:
            cf_saved = _save_uploads_to_tmp([cantus_firmus], tmp_dir)[0]
            m21_score = load_midi_files([str(cf_saved), str(cp_saved)])
            for idx, part in enumerate(m21_score.parts):
                part.partName = "cantus_firmus" if idx == 0 else "counterpoint"

        detected_key = parse_key(key) if key else detect_key(m21_score)
        internal = normalize_score(m21_score, key=detected_key)
        slices = extract_slices(internal)
        _annotate_slice_degrees(slices, detected_key)

        issues = run_species(
            internal,
            slices,
            species=species_num,
            params={
                "cantus_firmus_part": "cantus_firmus",
                "counterpoint_part": "counterpoint",
            },
        )
        result = AnalysisResult(metadata=internal.metadata, slices=slices, issues=issues)

        xml_path = tmp_dir / "score.musicxml"
        m21_score.write("musicxml", fp=xml_path)
        musicxml = xml_path.read_text()

    return JSONResponse(
        content=_sanitize_for_json({
            "result": result.model_dump(),
            "musicxml": musicxml,
            "species": species_num,
        })
    )


@app.post("/species-tutor")
async def species_tutor_endpoint(
    counterpoint: Annotated[UploadFile, File(...)],
    cantus_firmus: Annotated[UploadFile | None, File()] = None,
    preset: Annotated[str | None, Form()] = None,
    species_num: Annotated[int, Form()] = 1,
    key: Annotated[str | None, Form()] = None,
    _: str = Depends(_basic_auth),
) -> JSONResponse:
    """Run species check + ask Claude tutor for feedback."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503, detail="ANTHROPIC_API_KEY is not configured on the server"
        )
    if cantus_firmus is None and not preset:
        raise HTTPException(
            status_code=400,
            detail="Either cantus_firmus upload or preset name is required",
        )

    async with HEAVY_LOCK:
        payload = await _species_impl(counterpoint, cantus_firmus, preset, species_num, key)
        body = payload.body.decode()
        import json as _json
        data = _json.loads(body)
        result = AnalysisResult.model_validate(data["result"])
        feedback = critique_species(result, species=species_num)
        data["tutor_feedback"] = feedback
    return JSONResponse(content=data)


@app.get("/species-presets")
def species_presets(_: str = Depends(_basic_auth)) -> JSONResponse:
    return JSONResponse(
        content={
            name: {"key": p.key, "description": p.description, "notes": p.notes}
            for name, p in CF_PRESETS.items()
        }
    )


def _render_cf(name: str, fmt: str) -> bytes:
    """Render a built-in cantus firmus preset to MIDI or MusicXML bytes."""
    if name not in CF_PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown preset {name!r}")
    cf_part = CF_PRESETS[name].to_part(part_name="cantus_firmus")
    score = m21.stream.Score()
    score.insert(0, cf_part)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"cf.{fmt}"
        if fmt == "mid":
            score.write("midi", fp=path)
        elif fmt == "musicxml":
            score.write("musicxml", fp=path)
        else:
            raise HTTPException(status_code=400, detail="format must be mid or musicxml")
        return path.read_bytes()


@app.get("/cantus-firmus/{name}.mid")
def cantus_firmus_mid(name: str, _: str = Depends(_basic_auth)) -> Response:
    """Download a built-in cantus firmus as a Standard MIDI File."""
    data = _render_cf(name, "mid")
    return Response(
        content=data,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{name}.mid"'},
    )


@app.get("/cantus-firmus/{name}.musicxml")
def cantus_firmus_musicxml(name: str, _: str = Depends(_basic_auth)) -> Response:
    """Return a built-in cantus firmus as MusicXML for browser-side rendering."""
    data = _render_cf(name, "musicxml")
    return Response(
        content=data,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": f'inline; filename="{name}.musicxml"'},
    )


@app.post("/fix")
async def fix_endpoint(
    files: Annotated[list[UploadFile], File(...)],
    key: Annotated[str | None, Form()] = None,
    use_llm: Annotated[bool, Form()] = False,
    _: str = Depends(_basic_auth),
) -> Response:
  async with HEAVY_LOCK:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        saved = _save_uploads_to_tmp(files, tmp_dir)
        m21_score = load_midi_files([str(p) for p in saved])
        detected_key = _resolve_key(m21_score, key)
        result = _build_result(m21_score, detected_key, None)

        fix_dir = tmp_dir / "fix_output"
        fixes = propose_rule_fixes(
            normalize_score(m21_score, key=detected_key), result
        )
        if use_llm:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise HTTPException(
                    status_code=503,
                    detail="ANTHROPIC_API_KEY required for use_llm=true",
                )
            internal = normalize_score(m21_score, key=detected_key)
            fixes.extend(propose_llm_fixes(internal, result))

        written = apply_fixes_to_midi(m21_score, fixes, fix_dir)
        report = fix_dir / "fixes.txt"
        write_diff_report(fixes, report)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in written:
                zf.write(path, arcname=path.name)
            if report.exists():
                zf.write(report, arcname=report.name)
        zip_buf.seek(0)

    return Response(
        content=zip_buf.read(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="fix_output.zip"',
            "X-Fix-Count": str(len(fixes)),
        },
    )


INDEX_HTML = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>composition-advisor</title>
<style>
  body { font-family: -apple-system, "Hiragino Sans", sans-serif; max-width: 960px; margin: 1.5em auto; padding: 0 1em; color: #1f2937; }
  h1 { font-size: 1.4em; margin-bottom: 0.2em; }
  h2 { font-size: 1.05em; color: #475569; margin-top: 1.5em; }
  .tabs { display: flex; gap: 0.4em; border-bottom: 1px solid #cbd5e1; margin-bottom: 1em; }
  .tabs button { background: transparent; border: 0; padding: 0.6em 1em; cursor: pointer; color: #64748b; font-size: 0.95em; border-bottom: 3px solid transparent; }
  .tabs button.active { color: #2563eb; border-bottom-color: #2563eb; font-weight: 600; }
  .panel { display: none; }
  .panel.active { display: block; }
  form { border: 1px solid #cbd5e1; padding: 1em 1.5em; border-radius: 8px; margin-bottom: 1em; background: #fff; }
  label { display: block; margin: 0.6em 0; font-size: 0.92em; }
  input[type="text"], select { width: 100%; padding: 0.45em; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 0.95em; }
  button { padding: 0.55em 1.1em; background: #2563eb; color: white; border: 0; border-radius: 4px; cursor: pointer; font-size: 0.92em; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #94a3b8; cursor: not-allowed; }
  button.secondary { background: #64748b; }
  button.secondary:hover { background: #475569; }
  pre { background: #f1f5f9; padding: 1em; overflow-x: auto; max-height: 400px; border-radius: 6px; font-size: 0.85em; }
  .row { display: flex; gap: 0.5em; align-items: center; flex-wrap: wrap; }

  .dropzone {
    border: 2px dashed #94a3b8;
    border-radius: 8px;
    padding: 1.5em 1em;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  .dropzone.drag { background: #dbeafe; border-color: #2563eb; color: #1d4ed8; }
  .dropzone .hint { font-size: 0.85em; margin-top: 0.4em; color: #94a3b8; }

  .file-list { list-style: none; padding: 0; margin: 0.6em 0 0; font-size: 0.9em; }
  .file-list li {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3em 0.6em;
    background: #eff6ff;
    border-radius: 4px;
    margin-bottom: 0.25em;
  }
  .file-list button {
    background: transparent;
    color: #ef4444;
    padding: 0 0.4em;
    font-size: 1.1em;
  }

  .issues { margin-top: 0.5em; }
  .issue {
    border-left: 3px solid #cbd5e1;
    padding: 0.4em 0.8em;
    margin-bottom: 0.4em;
    background: #f8fafc;
    border-radius: 0 4px 4px 0;
    font-size: 0.88em;
  }
  .issue.warning { border-left-color: #f59e0b; }
  .issue.error { border-left-color: #ef4444; }
  .issue.info { border-left-color: #3b82f6; }
  .issue .head { font-weight: 600; color: #1e293b; }
  .issue .head .pos { color: #64748b; font-weight: 400; margin-left: 0.4em; font-family: ui-monospace, monospace; }

  #score-host {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 0.5em;
    background: #fff;
    min-height: 200px;
    overflow-x: auto;
  }
  #score-host:empty::before {
    content: "譜面はここに表示されます";
    color: #94a3b8;
    font-size: 0.85em;
  }
  .tutor-box { background: #fef3c7; border: 1px solid #fde68a; border-radius: 6px; padding: 1em; white-space: pre-wrap; font-size: 0.92em; line-height: 1.6; }
</style>
</head>
<body>
<h1>composition-advisor</h1>

<div class="tabs">
  <button data-tab="general" class="active">一般分析</button>
  <button data-tab="species">対位法レッスン</button>
</div>

<!-- ============ 一般分析 ============ -->
<section id="panel-general" class="panel active">
<form id="general-form" enctype="multipart/form-data">
  <div id="general-dropzone" class="dropzone" data-target="general">
    <div><strong>MIDI ファイルをここにドラッグ&ドロップ</strong></div>
    <div class="hint">またはクリックして選択(.mid / .midi、複数可)</div>
    <input id="general-file-input" type="file" name="files" multiple accept=".mid,.midi" hidden>
    <ul id="general-file-list" class="file-list"></ul>
  </div>

  <label>キー(省略時は自動推定)
    <input type="text" name="key" placeholder="C, Am, Bb …">
  </label>
  <div class="row">
    <button type="submit" data-mode="analyze">分析</button>
    <button type="submit" data-mode="critique">添削(Claude)</button>
    <button type="submit" data-mode="fix">修正MIDI</button>
  </div>
</form>
</section>

<!-- ============ 対位法レッスン ============ -->
<section id="panel-species" class="panel">
<form id="species-form" enctype="multipart/form-data">
  <label>Species
    <select name="species_num">
      <option value="1">Species 1 (1:1, note against note)</option>
      <option value="2">Species 2 (2:1, half notes)</option>
      <option value="3">Species 3 (4:1, quarter notes)</option>
      <option value="4">Species 4 (suspension)</option>
      <option value="5">Species 5 (florid)</option>
    </select>
  </label>

  <label>Cantus firmus(プリセット or アップロード)
    <select name="preset" id="species-preset">
      <option value="">(アップロードする)</option>
    </select>
  </label>

  <div id="cf-preview" style="display:none; border:1px solid #cbd5e1; border-radius:6px; padding:0.6em; background:#fff; margin-bottom:0.8em;">
    <div class="row" style="margin-bottom:0.4em;">
      <strong id="cf-preview-name" style="flex:1"></strong>
      <button type="button" id="cf-play" class="secondary">▶ 再生</button>
      <button type="button" id="cf-stop" class="secondary" style="display:none">■ 停止</button>
      <a id="cf-download" class="secondary" style="text-decoration:none; padding:0.55em 1.1em; background:#64748b; color:#fff; border-radius:4px; font-size:0.92em;" download>↓ MIDI</a>
    </div>
    <div id="cf-preview-host" style="min-height:120px;"></div>
    <div id="cf-preview-meta" style="font-size:0.8em; color:#64748b; margin-top:0.4em;"></div>
  </div>

  <div id="species-cf-dropzone" class="dropzone" data-target="species-cf" style="display:none">
    <div><strong>Cantus Firmus MIDI をドロップ</strong></div>
    <div class="hint">または クリック</div>
    <input id="species-cf-input" type="file" accept=".mid,.midi" hidden>
    <ul id="species-cf-list" class="file-list"></ul>
  </div>

  <h2>Counterpoint(あなたの解答)</h2>
  <div id="species-cp-dropzone" class="dropzone" data-target="species-cp">
    <div><strong>Counterpoint MIDI をドロップ</strong></div>
    <div class="hint">または クリック</div>
    <input id="species-cp-input" type="file" accept=".mid,.midi" hidden>
    <ul id="species-cp-list" class="file-list"></ul>
  </div>

  <label>キー(省略時は自動推定)
    <input type="text" name="key" placeholder="C, Am, Bb …">
  </label>

  <div class="row">
    <button type="submit" data-mode="species">チェック</button>
    <button type="submit" data-mode="species-tutor">添削(Claude教師)</button>
  </div>
</form>
</section>

<h2>譜面</h2>
<div id="score-host"></div>

<h2>Issues</h2>
<div id="issues" class="issues"></div>

<h2 id="tutor-heading" style="display:none">対位法教師フィードバック</h2>
<div id="tutor" class="tutor-box" style="display:none"></div>

<h2>生レスポンス</h2>
<pre id="output">結果がここに出ます</pre>

<script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.7/build/opensheetmusicdisplay.min.js"></script>
<script>
// ---------- common helpers ----------
function $(sel) { return document.querySelector(sel); }

const tabs = document.querySelectorAll('.tabs button');
const panels = document.querySelectorAll('.panel');
tabs.forEach((t) => {
  t.addEventListener('click', () => {
    tabs.forEach((x) => x.classList.remove('active'));
    panels.forEach((p) => p.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('panel-' + t.dataset.tab).classList.add('active');
  });
});

// ---------- Multi-file dropzone factory ----------
function bindDropzone(zoneId, inputId, listId, multiple) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const list = document.getElementById(listId);
  let files = [];

  function refresh() {
    list.innerHTML = '';
    files.forEach((f, idx) => {
      const li = document.createElement('li');
      const span = document.createElement('span');
      span.textContent = f.name + ' (' + Math.round(f.size / 1024) + ' KB)';
      const rm = document.createElement('button');
      rm.type = 'button';
      rm.textContent = '×';
      rm.title = '削除';
      rm.addEventListener('click', (e) => {
        e.stopPropagation();
        files.splice(idx, 1);
        refresh();
      });
      li.appendChild(span);
      li.appendChild(rm);
      list.appendChild(li);
    });
  }
  function add(newFiles) {
    for (const f of newFiles) {
      const lower = f.name.toLowerCase();
      if (!lower.endsWith('.mid') && !lower.endsWith('.midi')) continue;
      if (files.some((x) => x.name === f.name && x.size === f.size)) continue;
      if (!multiple) files = [];
      files.push(f);
      if (!multiple) break;
    }
    refresh();
  }
  zone.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') return;
    input.click();
  });
  input.addEventListener('change', () => add(input.files));
  ['dragenter', 'dragover'].forEach((ev) => {
    zone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      zone.classList.add('drag');
    });
  });
  ['dragleave', 'drop'].forEach((ev) => {
    zone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      zone.classList.remove('drag');
    });
  });
  zone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files) add(e.dataTransfer.files);
  });
  return {
    files: () => files,
    clear: () => { files = []; refresh(); },
  };
}

['dragover', 'drop'].forEach((ev) => {
  window.addEventListener(ev, (e) => e.preventDefault());
});

const generalDz = bindDropzone('general-dropzone', 'general-file-input', 'general-file-list', true);
const speciesCfDz = bindDropzone('species-cf-dropzone', 'species-cf-input', 'species-cf-list', false);
const speciesCpDz = bindDropzone('species-cp-dropzone', 'species-cp-input', 'species-cp-list', false);

// ---------- preset loader ----------
let presetData = {};
async function loadPresets() {
  try {
    const res = await fetch('/species-presets');
    if (!res.ok) return;
    presetData = await res.json();
    const sel = document.getElementById('species-preset');
    for (const name of Object.keys(presetData)) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name + ' (' + presetData[name].key + ')';
      sel.appendChild(opt);
    }
  } catch (err) { console.warn('preset load failed', err); }
}
loadPresets();

// ---------- cantus firmus preview ----------
let cfOsmd = null;
async function showCfPreview(name) {
  const preview = document.getElementById('cf-preview');
  if (!name) {
    preview.style.display = 'none';
    return;
  }
  preview.style.display = '';
  const meta = presetData[name];
  document.getElementById('cf-preview-name').textContent = name + ' (' + meta.key + ')';
  document.getElementById('cf-preview-meta').textContent =
    meta.description + '  /  音: ' + meta.notes.join(' ');
  document.getElementById('cf-download').href = '/cantus-firmus/' + encodeURIComponent(name) + '.mid';

  // Fetch MusicXML and draw it.
  try {
    const res = await fetch('/cantus-firmus/' + encodeURIComponent(name) + '.musicxml');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const xml = await res.text();
    const host = document.getElementById('cf-preview-host');
    host.innerHTML = '';
    if (!cfOsmd || cfOsmd._host !== host) {
      cfOsmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(host, {
        autoResize: true, drawTitle: false, drawSubtitle: false, drawComposer: false,
      });
      cfOsmd._host = host;
    }
    await cfOsmd.load(xml);
    cfOsmd.render();
  } catch (err) {
    document.getElementById('cf-preview-host').textContent = 'プレビュー失敗: ' + err.message;
  }
}

document.getElementById('species-preset').addEventListener('change', (e) => {
  const val = e.target.value;
  document.getElementById('species-cf-dropzone').style.display = val ? 'none' : '';
  showCfPreview(val);
});

// ---------- WebAudio playback for cantus firmus preview ----------
// Studio One pitch labelling: middle C = C3 = MIDI 60.
const NOTE_BASE = { C:0, D:2, E:4, F:5, G:7, A:9, B:11 };
function studioOneToMidi(name) {
  const m = /^([A-Ga-g])([#b]?)(-?\d+)$/.exec(name.trim());
  if (!m) return null;
  let v = NOTE_BASE[m[1].toUpperCase()];
  if (m[2] === '#') v += 1;
  if (m[2] === 'b') v -= 1;
  return v + (parseInt(m[3], 10) + 2) * 12;
}
function midiToFreq(midi) { return 440 * Math.pow(2, (midi - 69) / 12); }

let audioCtx = null;
let cfPlayback = null;
function stopCfPlayback() {
  if (cfPlayback) {
    cfPlayback.forEach((node) => { try { node.stop(); } catch {} });
    cfPlayback = null;
  }
  document.getElementById('cf-stop').style.display = 'none';
  document.getElementById('cf-play').style.display = '';
}
function playCf(name) {
  const meta = presetData[name];
  if (!meta) return;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  stopCfPlayback();

  const beatDur = 0.6;  // seconds per quarter; whole note ~ 2.4s
  const noteLen = beatDur * 4;
  const now = audioCtx.currentTime + 0.05;
  const created = [];
  meta.notes.forEach((nm, i) => {
    const midi = studioOneToMidi(nm);
    if (midi == null) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'triangle';
    osc.frequency.value = midiToFreq(midi);
    osc.connect(gain).connect(audioCtx.destination);
    const start = now + i * noteLen;
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.18, start + 0.02);
    gain.gain.setValueAtTime(0.18, start + noteLen * 0.85);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + noteLen);
    osc.start(start);
    osc.stop(start + noteLen + 0.05);
    created.push(osc);
  });
  cfPlayback = created;
  document.getElementById('cf-stop').style.display = '';
  document.getElementById('cf-play').style.display = 'none';
  // Auto-stop after total length
  setTimeout(stopCfPlayback, (meta.notes.length * noteLen + 0.5) * 1000);
}
document.getElementById('cf-play').addEventListener('click', () => {
  const name = document.getElementById('species-preset').value;
  if (name) playCf(name);
});
document.getElementById('cf-stop').addEventListener('click', stopCfPlayback);

// ---------- OSMD renderer ----------
let osmd = null;
async function renderMusicXML(xml, issues) {
  const host = document.getElementById('score-host');
  host.innerHTML = '';
  if (!xml) return;
  if (!osmd || osmd._host !== host) {
    osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(host, {
      autoResize: true, drawTitle: false, drawSubtitle: false, drawComposer: false,
    });
    osmd._host = host;
  }
  try {
    await osmd.load(xml);
    osmd.render();
    highlightIssues(issues);
  } catch (err) {
    host.textContent = '譜面の描画に失敗: ' + err.message;
  }
}

function highlightIssues(issues) {
  if (!osmd || !issues) return;
  // OSMD 1.x: traverse measures and color the first staff entry of bars
  // that have issues. We don't try to pinpoint exact beats — too brittle.
  const bars = new Map();
  issues.forEach((iss) => {
    const bucket = bars.get(iss.bar) || [];
    bucket.push(iss);
    bars.set(iss.bar, bucket);
  });
  try {
    const sheet = osmd.Sheet;
    const measureList = sheet && sheet.SourceMeasures;
    if (!measureList) return;
    measureList.forEach((measure) => {
      const num = measure.MeasureNumber;
      if (!bars.has(num)) return;
      const issuesHere = bars.get(num);
      const worst = issuesHere.reduce((acc, x) => {
        const order = { info: 1, warning: 2, error: 3 };
        return order[x.severity] > order[acc.severity] ? x : acc;
      });
      const color = worst.severity === 'error' ? '#ef4444'
                  : worst.severity === 'warning' ? '#f59e0b'
                  : '#3b82f6';
      measure.VerticalSourceStaffEntryContainers.forEach((vsse) => {
        vsse.StaffEntries.forEach((entry) => {
          if (!entry) return;
          entry.VoiceEntries.forEach((ve) => {
            ve.Notes.forEach((note) => {
              if (note && note.PrintObject !== false) {
                note.NoteheadColor = color;
              }
            });
          });
        });
      });
    });
    osmd.render();
  } catch (err) { console.warn('highlight failed', err); }
}

// ---------- issue list renderer ----------
function renderIssues(issues) {
  const host = document.getElementById('issues');
  host.innerHTML = '';
  if (!issues || !issues.length) {
    host.innerHTML = '<div class="issue info"><div class="head">問題なし</div></div>';
    return;
  }
  issues.forEach((iss) => {
    const div = document.createElement('div');
    div.className = 'issue ' + iss.severity;
    const head = document.createElement('div');
    head.className = 'head';
    head.textContent = iss.rule_id;
    const pos = document.createElement('span');
    pos.className = 'pos';
    pos.textContent = ' bar' + iss.bar + ' beat' + Number(iss.beat_in_bar).toFixed(2);
    head.appendChild(pos);
    const desc = document.createElement('div');
    desc.textContent = iss.description;
    div.appendChild(head);
    div.appendChild(desc);
    host.appendChild(div);
  });
}

// ---------- shared submit helper ----------
async function postJSON(url, fd) {
  const res = await fetch(url, { method: 'POST', body: fd });
  const text = await res.text();
  if (!res.ok) throw new Error('HTTP ' + res.status + ': ' + text);
  return JSON.parse(text);
}

function setBusy(form, busy) {
  form.querySelectorAll('button[type="submit"]').forEach((b) => (b.disabled = busy));
}

// ---------- general form submit ----------
document.getElementById('general-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const files = generalDz.files();
  if (!files.length) { alert('MIDI ファイルを選んでください'); return; }
  const mode = e.submitter.dataset.mode;
  const form = e.currentTarget;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  const key = form.elements['key'].value;
  if (key) fd.append('key', key);

  setBusy(form, true);
  document.getElementById('output').textContent = '送信中…';
  try {
    if (mode === 'fix') {
      const res = await fetch('/fix', { method: 'POST', body: fd });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'fix_output.zip'; a.click();
      document.getElementById('output').textContent = 'fix_output.zip をダウンロードしました (' + res.headers.get('X-Fix-Count') + ' fixes)';
      return;
    }

    if (mode === 'analyze') {
      // 1) get analysis JSON  2) get musicxml separately for rendering
      const data = await postJSON('/analyze', fd);
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      renderIssues(data.issues);

      const fdXml = new FormData();
      for (const f of files) fdXml.append('files', f);
      const xmlRes = await fetch('/musicxml', { method: 'POST', body: fdXml });
      if (xmlRes.ok) await renderMusicXML(await xmlRes.text(), data.issues);
      return;
    }

    if (mode === 'critique') {
      const data = await postJSON('/critique', fd);
      document.getElementById('output').textContent = data.critique;
      const tutor = document.getElementById('tutor');
      const heading = document.getElementById('tutor-heading');
      tutor.style.display = ''; heading.style.display = '';
      tutor.textContent = data.critique;
      return;
    }
  } catch (err) {
    document.getElementById('output').textContent = 'エラー: ' + err.message;
  } finally {
    setBusy(form, false);
  }
});

// ---------- species form submit ----------
document.getElementById('species-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const cpFiles = speciesCpDz.files();
  if (!cpFiles.length) { alert('Counterpoint の MIDI を選んでください'); return; }
  const presetVal = form.elements['preset'].value;
  const cfFiles = speciesCfDz.files();
  if (!presetVal && !cfFiles.length) { alert('Cantus firmus を選ぶか、プリセットを選んでください'); return; }

  const mode = e.submitter.dataset.mode;  // 'species' or 'species-tutor'
  const fd = new FormData();
  fd.append('counterpoint', cpFiles[0]);
  if (presetVal) fd.append('preset', presetVal);
  else fd.append('cantus_firmus', cfFiles[0]);
  fd.append('species_num', form.elements['species_num'].value);
  const key = form.elements['key'].value;
  if (key) fd.append('key', key);

  setBusy(form, true);
  document.getElementById('output').textContent = '送信中…(species check)';
  document.getElementById('tutor').style.display = 'none';
  document.getElementById('tutor-heading').style.display = 'none';

  try {
    const data = await postJSON('/' + mode, fd);
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    const result = data.result;
    renderIssues(result.issues);
    if (data.musicxml) await renderMusicXML(data.musicxml, result.issues);
    if (data.tutor_feedback) {
      document.getElementById('tutor').style.display = '';
      document.getElementById('tutor-heading').style.display = '';
      document.getElementById('tutor').textContent = data.tutor_feedback;
    }
  } catch (err) {
    document.getElementById('output').textContent = 'エラー: ' + err.message;
  } finally {
    setBusy(form, false);
  }
});
</script>
</body>
</html>
"""
