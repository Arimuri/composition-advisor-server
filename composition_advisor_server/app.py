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

import io
import logging
import os
import secrets
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import music21 as m21

from composition_advisor.analyze.key_detector import detect_key, parse_key
from composition_advisor.analyze.voice_extractor import extract_slices
from composition_advisor.cli import _annotate_slice_degrees
from composition_advisor.critique.config import load_config
from composition_advisor.critique.runner import run_all as run_all_rules
from composition_advisor.fix.applier import apply_fixes_to_midi, write_diff_report
from composition_advisor.fix.llm import propose as propose_llm_fixes
from composition_advisor.fix.rule_based import propose as propose_rule_fixes
from composition_advisor.io.midi_loader import load_midi_files
from composition_advisor.io.normalize import normalize_score
from composition_advisor.llm.claude_client import critique as llm_critique
from composition_advisor.model.issue import AnalysisResult

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
    """Persist uploaded files to a temp directory and return their paths."""
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
        with out_path.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        saved.append(out_path)
    if not saved:
        raise HTTPException(status_code=400, detail="No MIDI files uploaded")
    return saved


def _resolve_key(m21_score: m21.stream.Score, key_arg: str | None) -> m21.key.Key:
    return parse_key(key_arg) if key_arg else detect_key(m21_score)


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
def analyze_endpoint(
    files: Annotated[list[UploadFile], File(...)],
    key: Annotated[str | None, Form()] = None,
    _: str = Depends(_basic_auth),
) -> JSONResponse:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        saved = _save_uploads_to_tmp(files, tmp_dir)
        m21_score = load_midi_files([str(p) for p in saved])
        detected_key = _resolve_key(m21_score, key)
        result = _build_result(m21_score, detected_key, None)
    return JSONResponse(content=result.model_dump())


@app.post("/critique")
def critique_endpoint(
    files: Annotated[list[UploadFile], File(...)],
    key: Annotated[str | None, Form()] = None,
    _: str = Depends(_basic_auth),
) -> JSONResponse:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503, detail="ANTHROPIC_API_KEY is not configured on the server"
        )
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


@app.post("/fix")
def fix_endpoint(
    files: Annotated[list[UploadFile], File(...)],
    key: Annotated[str | None, Form()] = None,
    use_llm: Annotated[bool, Form()] = False,
    _: str = Depends(_basic_auth),
) -> Response:
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


INDEX_HTML = """\
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>composition-advisor</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 2em auto; padding: 0 1em; }
  h1 { font-size: 1.4em; }
  form { border: 1px solid #ccc; padding: 1em 1.5em; border-radius: 6px; margin-bottom: 1.5em; }
  label { display: block; margin: 0.6em 0; }
  input[type="text"] { width: 100%; padding: 0.4em; box-sizing: border-box; }
  button { padding: 0.6em 1.2em; background: #2563eb; color: white; border: 0; border-radius: 4px; cursor: pointer; font-size: 0.95em; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #94a3b8; cursor: not-allowed; }
  pre { background: #f4f4f4; padding: 1em; overflow-x: auto; max-height: 480px; }
  .row { display: flex; gap: 0.5em; align-items: center; flex-wrap: wrap; }

  .dropzone {
    border: 2px dashed #94a3b8;
    border-radius: 8px;
    padding: 2em 1em;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  .dropzone.drag {
    background: #dbeafe;
    border-color: #2563eb;
    color: #1d4ed8;
  }
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
  .file-list button:hover { background: transparent; color: #b91c1c; }
</style>
</head>
<body>
<h1>composition-advisor</h1>

<form id="analyze" enctype="multipart/form-data">
  <div id="dropzone" class="dropzone">
    <div><strong>MIDI ファイルをここにドラッグ&ドロップ</strong></div>
    <div class="hint">またはクリックして選択(.mid / .midi、複数可)</div>
    <input id="file-input" type="file" name="files" multiple accept=".mid,.midi" hidden>
    <ul id="file-list" class="file-list"></ul>
  </div>

  <label>キー(省略時は自動推定)
    <input type="text" name="key" placeholder="C, Am, Bb …">
  </label>
  <div class="row">
    <button type="submit" data-mode="analyze">分析(JSON)</button>
    <button type="submit" data-mode="critique">添削(Claude)</button>
    <button type="submit" data-mode="fix">修正MIDI</button>
  </div>
</form>

<pre id="output">結果がここに出ます</pre>

<script>
const form = document.getElementById('analyze');
const out = document.getElementById('output');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');

// In-memory list because DataTransfer.files cannot be mutated cross-browser.
let selectedFiles = [];

function refreshList() {
  fileList.innerHTML = '';
  selectedFiles.forEach((f, idx) => {
    const li = document.createElement('li');
    const span = document.createElement('span');
    span.textContent = f.name + ' (' + Math.round(f.size / 1024) + ' KB)';
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.textContent = '×';
    rm.title = '削除';
    rm.addEventListener('click', (e) => {
      e.stopPropagation();
      selectedFiles.splice(idx, 1);
      refreshList();
    });
    li.appendChild(span);
    li.appendChild(rm);
    fileList.appendChild(li);
  });
}

function addFiles(files) {
  for (const f of files) {
    const lower = f.name.toLowerCase();
    if (!lower.endsWith('.mid') && !lower.endsWith('.midi')) continue;
    // Skip duplicates by name+size.
    if (selectedFiles.some((x) => x.name === f.name && x.size === f.size)) continue;
    selectedFiles.push(f);
  }
  refreshList();
}

dropzone.addEventListener('click', (e) => {
  if (e.target.tagName === 'BUTTON') return;
  fileInput.click();
});
fileInput.addEventListener('change', () => addFiles(fileInput.files));

['dragenter', 'dragover'].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add('drag');
  });
});
['dragleave', 'drop'].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('drag');
  });
});
dropzone.addEventListener('drop', (e) => {
  if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
});

// Prevent the browser from navigating away on accidental drops outside the zone.
['dragover', 'drop'].forEach((ev) => {
  window.addEventListener(ev, (e) => e.preventDefault());
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (selectedFiles.length === 0) {
    out.textContent = 'MIDI ファイルを選んでください';
    return;
  }
  const mode = e.submitter.dataset.mode;
  const fd = new FormData();
  for (const f of selectedFiles) fd.append('files', f);
  const key = form.elements['key'].value;
  if (key) fd.append('key', key);

  out.textContent = '送信中…';
  const buttons = form.querySelectorAll('button[type="submit"]');
  buttons.forEach((b) => (b.disabled = true));
  try {
    const res = await fetch('/' + mode, { method: 'POST', body: fd });
    if (mode === 'fix' && res.ok) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'fix_output.zip';
      a.click();
      out.textContent = 'fix_output.zip をダウンロードしました (' + res.headers.get('X-Fix-Count') + ' fixes)';
      return;
    }
    const text = await res.text();
    if (!res.ok) {
      out.textContent = 'HTTP ' + res.status + '\\n' + text;
      return;
    }
    try { out.textContent = JSON.stringify(JSON.parse(text), null, 2); }
    catch { out.textContent = text; }
  } catch (err) {
    out.textContent = 'エラー: ' + err.message;
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
});
</script>
</body>
</html>
"""
