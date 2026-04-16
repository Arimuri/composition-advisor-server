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
from composition_advisor.analyze.note_annotations import annotate_score
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
from composition_advisor.tutor.lesson_runner import (
    build_lesson_system_prompt,
    build_lesson_user_prompt,
    run_lesson,
)
from composition_advisor.tutor.tracks import get_registry as get_track_registry

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


_HARMONIC_LABELS = {
    0: "P1",   # unison
    1: "m2",
    2: "M2",
    3: "m3",
    4: "M3",
    5: "P4",
    6: "TT",
    7: "P5",
    8: "m6",
    9: "M6",
    10: "m7",
    11: "M7",
}


def _harmonic_interval_label(low_midi: int, high_midi: int) -> str:
    """Return a textbook harmonic-interval label (e.g. 'P5', 'm3', 'M6')."""
    diff = abs(high_midi - low_midi)
    if diff == 0:
        return "P1"
    octaves = diff // 12
    base = _HARMONIC_LABELS[diff % 12]
    if octaves == 0:
        return base
    # Simple compound notation: P8 instead of P1+1oct
    if diff == 12:
        return "P8"
    if diff % 12 == 0:
        return f"P{octaves * 7 + 1}"  # 15, 22 ...
    return f"{base}+{octaves}oct"


def _attach_lyrics(m21_score: m21.stream.Score, key: m21.key.Key) -> None:
    """Annotate each note with scale degree + harmonic interval against cf.

    Layout:
        line 1 — scale degree of THIS note relative to the song key
                 (e.g. "3" for E in C major, "♯4" for F# in C major).
        line 2 — counterpoint only: harmonic interval between this note
                 and the cantus firmus note sounding at the same beat
                 (e.g. "5", "3", "♭7"). Cantus firmus notes get no
                 second line (it would just say "1" against itself).

    Lyric text never starts with "-" so music21's syllabic-hyphen
    handling does not silently strip characters.
    """
    from composition_advisor.analyze.note_annotations import DEGREE_LABELS

    tonic_pc = key.tonic.pitchClass if key is not None else None
    EPS = 1e-3

    # Find the cantus firmus part(s) for the score: any part whose name
    # contains "cantus". If none is found, skip the harmonic-interval row.
    cf_parts = [
        p for p in m21_score.parts
        if p.partName and "cantus" in p.partName.lower()
    ]

    def cf_midi_at(beat: float) -> int | None:
        """Return the cantus midi sounding at the given absolute beat."""
        for cp in cf_parts:
            for note in cp.flatten().notes:
                start = float(note.offset)
                end = start + float(note.duration.quarterLength)
                if start - EPS <= beat < end - EPS:
                    if isinstance(note, m21.chord.Chord):
                        return int(note.bass().midi)
                    return int(note.pitch.midi)
        return None

    for part in m21_score.parts:
        is_cf = part in cf_parts
        for elem in part.flatten().notes:
            try:
                if isinstance(elem, m21.chord.Chord):
                    midi = int(elem.bass().midi) if elem.pitches else None
                else:
                    midi = int(elem.pitch.midi)
            except Exception:
                continue
            if midi is None:
                continue

            lines: list[str] = []
            if tonic_pc is not None:
                offset = (midi - tonic_pc) % 12
                lines.append(DEGREE_LABELS[offset])

            if not is_cf and cf_parts:
                cf_midi = cf_midi_at(float(elem.offset))
                if cf_midi is not None:
                    lines.append(_harmonic_interval_label(cf_midi, midi))

            if lines:
                lyrics = []
                for i, t in enumerate(lines):
                    ly = m21.note.Lyric(text=t, number=i + 1, applyRaw=True)
                    lyrics.append(ly)
                elem.lyrics = lyrics


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
) -> tuple[AnalysisResult, list]:
    cfg = load_config(config_path) if config_path else None
    internal = normalize_score(m21_score, key=detected_key)
    slices = extract_slices(internal)
    _annotate_slice_degrees(slices, detected_key)
    issues = run_all_rules(internal, slices, config=cfg)
    annotations = annotate_score(internal, key=detected_key)
    result = AnalysisResult(metadata=internal.metadata, slices=slices, issues=issues)
    return result, annotations


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
            result, annotations = _build_result(m21_score, detected_key, None)
    payload = result.model_dump()
    payload["note_annotations"] = [a.model_dump() for a in annotations]
    return JSONResponse(content=_sanitize_for_json(payload))


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
            result, _annotations = _build_result(m21_score, detected_key, None)
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
    key: Annotated[str | None, Form()] = None,
    _: str = Depends(_basic_auth),
) -> Response:
    """Convert uploaded MIDI files into a single merged MusicXML.

    Used by the browser-side OSMD renderer to draw the score. The notes are
    annotated with scale-degree and melodic-interval lyrics so OSMD draws
    them automatically beneath each note.
    """
    async with HEAVY_LOCK:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            saved = _save_uploads_to_tmp(files, tmp_dir)
            m21_score = load_midi_files([str(p) for p in saved])
            detected_key = _resolve_key(m21_score, key)
            _attach_lyrics(m21_score, detected_key)
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
        annotations = annotate_score(internal, key=detected_key)

        _attach_lyrics(m21_score, detected_key)
        xml_path = tmp_dir / "score.musicxml"
        m21_score.write("musicxml", fp=xml_path)
        musicxml = xml_path.read_text()

    return JSONResponse(
        content=_sanitize_for_json({
            "result": result.model_dump(),
            "note_annotations": [a.model_dump() for a in annotations],
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


# ----- species rule descriptions(教科書的説明) ---------------------------

# 各 species のルールを「セクションごとの箇条書き」で持つ。UI 側はこれを
# そのまま表示する。本体側のルール ID とは別に、人間が読みやすい順序と
# 文言で書き下している(本体のルール名と1対1ではない場合あり)。

_SPECIES_RULES_JA: dict[int, dict[str, list[dict[str, str]]]] = {
    1: {
        "label": "Species 1 (1:1, note against note)",
        "summary": "定旋律1音に対して対旋律も1音(同じ長さ)。すべての音が同時に鳴り、不協和音は使えない。",
        "sections": [
            {
                "heading": "開始と終止",
                "items": [
                    "最初の音と最後の音は完全協和音(完全1度・完全5度・完全8度)で鳴らす",
                    "終止の直前は順次進行で目的音に到達するのが基本",
                ],
            },
            {
                "heading": "縦の音程(ハーモニック・インターバル)",
                "items": [
                    "すべての縦音程は協和音である必要がある(P1, m3, M3, P5, m6, M6, P8)",
                    "不協和音(m2, M2, P4, TT, m7, M7)は禁止",
                ],
            },
            {
                "heading": "旋律の動き",
                "items": [
                    "オクターブを超える跳躍は禁止",
                    "旋律的三全音(F→B など増4度・減5度の進行)は禁止",
                    "4度より大きな跳躍の後は反対方向への順次進行で「埋め戻す」",
                    "同方向への連続した跳躍は避ける",
                ],
            },
            {
                "heading": "旋律の形",
                "items": [
                    "クライマックス(最高音)は曲の中で1度だけ現れる",
                    "全体の音域は10度以内に収める",
                    "同じ高さの音を直接連打するのは禁止",
                ],
            },
            {
                "heading": "声部間の動き",
                "items": [
                    "平行5度・平行8度は禁止(2声部が同方向に動いて連続して完全音程を作る)",
                    "隠伏5度・隠伏8度は注意(両声部が同方向に動いて完全音程に到達する)",
                    "声部交叉(上声部が下声部より低くなる)は避ける",
                ],
            },
        ],
    },
    2: {
        "label": "Species 2 (2:1, half notes)",
        "summary": "定旋律1音に対して対旋律は2音(各2分音符)。強拍は協和必須、弱拍は経過音なら不協和も可。",
        "sections": [
            {
                "heading": "リズムと配置",
                "items": [
                    "1つの cantus 音に対して、対旋律は2分音符 × 2(強拍 + 弱拍)で書く",
                    "開始は休符からでもよい(その場合の最初の音は弱拍)",
                ],
            },
            {
                "heading": "強拍(downbeat)",
                "items": [
                    "強拍(cantus と同時に鳴るタイミング)は必ず協和音にする",
                    "禁則は1種と同じ(P1, m3, M3, P5, m6, M6, P8 のみ)",
                ],
            },
            {
                "heading": "弱拍(upbeat)",
                "items": [
                    "弱拍は協和音でも問題なし",
                    "弱拍が不協和になる場合は「経過音(passing tone)」のみ許可",
                    "経過音の条件: 直前の音から順次進行で接近し、直後の音にも同方向の順次進行で抜ける",
                    "上記以外の不協和(刺繍音を弱拍に置く等)は禁止",
                ],
            },
            {
                "heading": "1種から引き継ぐルール",
                "items": [
                    "開始・終止は完全協和音",
                    "平行5度・平行8度・声部交叉は禁止",
                    "旋律的三全音禁止、跳躍の後は順次進行で解決",
                    "クライマックスは1度だけ、音域10度以内",
                ],
            },
        ],
    },
    3: {
        "label": "Species 3 (4:1, quarter notes)",
        "summary": "定旋律1音に対して対旋律は4音(各4分音符)。経過音と刺繍音(neighbor)が登場する。",
        "sections": [
            {
                "heading": "リズムと配置",
                "items": [
                    "1つの cantus 音に対して、対旋律は4分音符 × 4 を書く",
                    "拍位置: 1拍目(最強拍)→ 2拍目(弱拍)→ 3拍目(準強拍)→ 4拍目(弱拍)",
                ],
            },
            {
                "heading": "1拍目(downbeat)",
                "items": [
                    "1拍目は必ず協和音にする(警告)",
                ],
            },
            {
                "heading": "3拍目(secondary strong)",
                "items": [
                    "3拍目は原則として協和音(info)",
                    "経過音または刺繍音として使う場合は許容",
                ],
            },
            {
                "heading": "2拍目・4拍目(weak)",
                "items": [
                    "経過音(stepwise approach + same-direction stepwise resolution)なら不協和も可",
                    "刺繍音(neighbor: a → b → a の形、b は a から半音 or 全音)も可",
                    "上記以外の不協和は避ける",
                ],
            },
            {
                "heading": "1種から引き継ぐルール",
                "items": [
                    "開始・終止は完全協和音",
                    "平行5度・平行8度・声部交叉は禁止",
                    "旋律的三全音禁止、跳躍の後は順次進行で解決",
                ],
            },
        ],
    },
    4: {
        "label": "Species 4 (suspension)",
        "summary": "対旋律は2分音符だが、弱拍と次の強拍をタイで結ぶ。掛留(サスペンション)の不協和→解決を体感する種。",
        "sections": [
            {
                "heading": "リズムと配置",
                "items": [
                    "対旋律は2分音符 × 2 だが、弱拍の音を次の小節の強拍にタイで持ち越す",
                    "結果として強拍に「前の音」が新しい cantus 音とぶつかる構造になる",
                ],
            },
            {
                "heading": "掛留の3段階",
                "items": [
                    "1. 準備 (preparation): タイの始点(弱拍)では協和音である必要がある",
                    "2. 衝突 (clash): タイの後半(次の強拍)では新 cantus 音に対して不協和になることが望ましい",
                    "3. 解決 (resolution): 次の弱拍で必ず下行2度(半音 or 全音)で解決し、協和音に到達",
                ],
            },
            {
                "heading": "代表的な掛留",
                "items": [
                    "7-6 サスペンション: 7度の不協和 → 6度の協和",
                    "4-3 サスペンション: 4度の不協和 → 3度の協和",
                    "9-8 サスペンション: 9度(または2度)→ オクターブ",
                ],
            },
            {
                "heading": "禁則",
                "items": [
                    "解決が下行2度以外(同音 or 上行 or 跳躍)になるのは禁止",
                    "解決音そのものが不協和なのは禁止",
                    "準備が不協和なのは禁止",
                ],
            },
            {
                "heading": "1種から引き継ぐルール",
                "items": [
                    "開始・終止は完全協和音",
                    "平行5度・平行8度・声部交叉は禁止",
                    "旋律的三全音禁止",
                ],
            },
        ],
    },
    5: {
        "label": "Species 5 (florid counterpoint)",
        "summary": "1〜4種を組み合わせて自由に書く華麗対位法。リズムは混在、掛留も使う。基礎ルールはすべて生きている。",
        "sections": [
            {
                "heading": "リズムの自由度",
                "items": [
                    "全音符・2分音符・4分音符・タイを自由に混在させてよい",
                    "ただし4音以上同じ音価が連続するのは単調になるので避ける",
                    "曲中に少なくとも1つは持続音や掛留を入れて変化を作る",
                ],
            },
            {
                "heading": "1〜4種から引き継ぐすべてのルール",
                "items": [
                    "強拍は協和音(1種・2種)",
                    "弱拍の不協和は経過音/刺繍音として(2種・3種)",
                    "掛留は準備→衝突→下行2度解決(4種)",
                    "開始・終止は完全協和音",
                    "平行5度・平行8度・隠伏5/8度・声部交叉に注意",
                    "旋律的三全音禁止、跳躍の後は順次進行で解決",
                    "クライマックスは1度だけ、音域10度以内",
                ],
            },
            {
                "heading": "華麗対位法ならではの目標",
                "items": [
                    "リズムの変化で旋律に「呼吸」と「物語」を作る",
                    "各拍ごとの細かい禁則よりも、全体としての歌いやすさを優先する",
                    "掛留を効果的なポイントで使い、最も印象的な瞬間を作る",
                ],
            },
        ],
    },
}


@app.get("/species-rules")
def species_rules(_: str = Depends(_basic_auth)) -> JSONResponse:
    """Return the textbook rules for each species (Japanese)."""
    return JSONResponse(content=_SPECIES_RULES_JA)


# ----- learning tracks (Phase F + G) ---------------------------------------

@app.get("/tracks")
def list_tracks(_: str = Depends(_basic_auth)) -> JSONResponse:
    """Return all learning tracks with their lesson summaries."""
    registry = get_track_registry()
    payload = {
        "tracks": [
            {
                "id": t.id,
                "title": t.title,
                "summary": t.summary,
                "lessons": [
                    {
                        "id": l.id,
                        "title": l.title,
                        "summary": l.summary,
                        "expected_parts": l.expected_parts,
                        "cantus_firmus_presets": l.cantus_firmus_presets,
                        "species_compat": l.species_compat,
                    }
                    for l in t.lessons
                ],
            }
            for t in registry.tracks.values()
        ]
    }
    return JSONResponse(content=payload)


@app.get("/tracks/{track_id}/lessons/{lesson_id}")
def get_lesson(track_id: str, lesson_id: str, _: str = Depends(_basic_auth)) -> JSONResponse:
    """Return one lesson's full definition (rules, persona, rule_card, references)."""
    registry = get_track_registry()
    lesson = registry.get_lesson(track_id, lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail=f"Unknown lesson {track_id}/{lesson_id}")
    return JSONResponse(content=lesson.model_dump())


@app.post("/lesson")
async def lesson_endpoint(
    track_id: Annotated[str, Form(...)],
    lesson_id: Annotated[str, Form(...)],
    counterpoint: Annotated[UploadFile, File(...)],
    cantus_firmus: Annotated[UploadFile | None, File()] = None,
    preset: Annotated[str | None, Form()] = None,
    key: Annotated[str | None, Form()] = None,
    use_llm: Annotated[bool, Form()] = False,
    _: str = Depends(_basic_auth),
) -> JSONResponse:
    """Run a lesson against the uploaded score."""
    registry = get_track_registry()
    lesson = registry.get_lesson(track_id, lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail=f"Unknown lesson {track_id}/{lesson_id}")

    # 2-part lessons (cantus firmus + counterpoint) accept either an uploaded
    # cf or one of the built-in presets, like /species. Multi-voice lessons
    # (3声/4声) require all the parts to be uploaded as a single MIDI file
    # passed in `counterpoint`(命名は雑だがここでは「main upload」という意味)。
    is_two_part = "cantus_firmus" in lesson.expected_parts
    async with HEAVY_LOCK:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cp_saved = _save_uploads_to_tmp([counterpoint], tmp_dir)[0]

            if is_two_part:
                if cantus_firmus is None and not preset:
                    raise HTTPException(
                        status_code=400,
                        detail="このレッスンは cantus firmus が必要です(プリセットかアップロード)。",
                    )
                if preset:
                    if preset not in CF_PRESETS:
                        raise HTTPException(status_code=400, detail=f"Unknown preset {preset!r}")
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
            else:
                # Multi-voice lessons: take the uploaded MIDI as-is.
                m21_score = load_midi_files([str(cp_saved)])
                # If the lesson expects specific part names, rename in order.
                if lesson.expected_parts:
                    for idx, part in enumerate(m21_score.parts):
                        if idx < len(lesson.expected_parts):
                            part.partName = lesson.expected_parts[idx]

            detected_key = parse_key(key) if key else detect_key(m21_score)
            internal = normalize_score(m21_score, key=detected_key)
            slices = extract_slices(internal)
            _annotate_slice_degrees(slices, detected_key)

            params = {}
            if is_two_part:
                params = {
                    "cantus_firmus_part": "cantus_firmus",
                    "counterpoint_part": "counterpoint",
                }
            issues = run_lesson(lesson, internal, slices, params=params)
            result = AnalysisResult(metadata=internal.metadata, slices=slices, issues=issues)
            annotations = annotate_score(internal, key=detected_key)

            _attach_lyrics(m21_score, detected_key)
            xml_path = tmp_dir / "score.musicxml"
            m21_score.write("musicxml", fp=xml_path)
            musicxml = xml_path.read_text()

            tutor_feedback = None
            if use_llm:
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise HTTPException(
                        status_code=503, detail="ANTHROPIC_API_KEY が設定されていません"
                    )
                # Build a lesson-specific Claude system prompt + base user prompt.
                from composition_advisor.llm.prompt_builder import build_user_prompt
                import anthropic as _anthropic
                base_prompt = build_user_prompt(result)
                user_prompt = build_lesson_user_prompt(lesson, base_prompt)
                system_prompt = build_lesson_system_prompt(lesson)
                client = _anthropic.Anthropic()
                resp = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                tutor_feedback = "\n".join(b.text for b in resp.content if hasattr(b, "text"))

    return JSONResponse(
        content=_sanitize_for_json({
            "lesson": lesson.model_dump(),
            "result": result.model_dump(),
            "note_annotations": [a.model_dump() for a in annotations],
            "musicxml": musicxml,
            "tutor_feedback": tutor_feedback,
        })
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
        result, _annotations = _build_result(m21_score, detected_key, None)

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
  .tutor-box { background: #fef3c7; border: 1px solid #fde68a; border-radius: 6px; padding: 1em; font-size: 0.92em; line-height: 1.6; }
  .tutor-box h1, .tutor-box h2, .tutor-box h3 { margin: 0.8em 0 0.3em; font-size: 1.05em; color: #92400e; }
  .tutor-box ul, .tutor-box ol { margin: 0.3em 0 0.3em 1.4em; }
  .tutor-box table { border-collapse: collapse; margin: 0.5em 0; font-size: 0.9em; }
  .tutor-box th, .tutor-box td { border: 1px solid #d97706; padding: 0.25em 0.5em; }
  .tutor-box code { background: #fef9c3; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }
</style>
</head>
<body>
<h1>composition-advisor</h1>

<div class="tabs">
  <button data-tab="general" class="active">一般分析</button>
  <button data-tab="species">対位法レッスン</button>
  <button data-tab="lesson">学習トラック</button>
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
    <select name="species_num" id="species-num">
      <option value="1">Species 1 (1:1, note against note)</option>
      <option value="2">Species 2 (2:1, half notes)</option>
      <option value="3">Species 3 (4:1, quarter notes)</option>
      <option value="4">Species 4 (suspension)</option>
      <option value="5">Species 5 (florid)</option>
    </select>
  </label>

  <div id="species-rules-card" style="border:1px solid #cbd5e1; border-radius:6px; padding:0.8em 1em; background:#fffdf5; margin:0.6em 0 0.8em;">
    <div style="font-weight:600; color:#475569; margin-bottom:0.3em;" id="species-rules-title">この種のルール</div>
    <div style="font-size:0.85em; color:#64748b; margin-bottom:0.6em;" id="species-rules-summary"></div>
    <div id="species-rules-body" style="font-size:0.85em; line-height:1.55;"></div>
  </div>

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
      <button type="button" id="cf-download" class="secondary">↓ MIDI</button>
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

<!-- ============ 学習トラック ============ -->
<section id="panel-lesson" class="panel">
<form id="lesson-form" enctype="multipart/form-data">
  <label>トラック
    <select name="track_id" id="lesson-track"></select>
  </label>
  <label>レッスン
    <select name="lesson_id" id="lesson-pick"></select>
  </label>

  <div id="lesson-meta" style="border:1px solid #cbd5e1; border-radius:6px; padding:0.8em 1em; background:#fffdf5; margin:0.6em 0;">
    <div style="font-weight:600; color:#475569;" id="lesson-title-display"></div>
    <div style="font-size:0.85em; color:#64748b; margin-top:0.3em;" id="lesson-summary-display"></div>
    <div style="font-size:0.8em; color:#94a3b8; margin-top:0.3em;" id="lesson-intent-display"></div>
    <div id="lesson-rule-card" style="font-size:0.85em; line-height:1.55; margin-top:0.5em;"></div>
    <div id="lesson-references" style="font-size:0.78em; color:#94a3b8; margin-top:0.5em;"></div>
  </div>

  <label id="lesson-cf-row">Cantus firmus(プリセット or アップロード)
    <select name="preset" id="lesson-preset">
      <option value="">(アップロードする)</option>
    </select>
  </label>

  <div id="lesson-cf-dropzone" class="dropzone" data-target="lesson-cf" style="display:none">
    <div><strong>Cantus Firmus MIDI をドロップ</strong></div>
    <div class="hint">または クリック</div>
    <input id="lesson-cf-input" type="file" accept=".mid,.midi" hidden>
    <ul id="lesson-cf-list" class="file-list"></ul>
  </div>

  <h2 id="lesson-cp-heading">提出する MIDI</h2>
  <div id="lesson-cp-dropzone" class="dropzone" data-target="lesson-cp">
    <div><strong>MIDI をドロップ</strong></div>
    <div class="hint">または クリック(複数声部のレッスンでは全声部を1ファイルに)</div>
    <input id="lesson-cp-input" type="file" accept=".mid,.midi" hidden>
    <ul id="lesson-cp-list" class="file-list"></ul>
  </div>

  <label>キー(省略時は自動推定)
    <input type="text" name="key" placeholder="C, Am, Bb …">
  </label>

  <div class="row">
    <button type="submit" data-mode="check">チェック</button>
    <button type="submit" data-mode="check-llm">チェック + Claude添削</button>
  </div>
</form>
</section>

<h2>譜面 <button type="button" id="score-play" class="secondary" style="font-size:0.85em; margin-left:0.5em;" disabled>▶ 再生</button><button type="button" id="score-stop" class="secondary" style="font-size:0.85em; margin-left:0.3em; display:none;">■ 停止</button></h2>
<div id="score-host"></div>

<h2>Issues</h2>
<div id="issues" class="issues"></div>

<h2 id="tutor-heading" style="display:none">対位法教師フィードバック</h2>
<div id="tutor" class="tutor-box" style="display:none"></div>

<h2>生レスポンス</h2>
<pre id="output">結果がここに出ます</pre>

<script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.7/build/opensheetmusicdisplay.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@15.0.7/lib/marked.umd.min.js"></script>
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

  function fmtSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  }

  function refresh() {
    list.innerHTML = '';
    files.forEach((f, idx) => {
      const li = document.createElement('li');
      const span = document.createElement('span');
      span.textContent = f.name + ' (' + fmtSize(f.size) + ')';
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
const lessonCfDz = bindDropzone('lesson-cf-dropzone', 'lesson-cf-input', 'lesson-cf-list', false);
const lessonCpDz = bindDropzone('lesson-cp-dropzone', 'lesson-cp-input', 'lesson-cp-list', false);

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
  document.getElementById('cf-download').dataset.cfName = name;

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

// ---------- species rules card ----------
let speciesRulesData = {};
async function loadSpeciesRules() {
  try {
    const res = await fetch('/species-rules');
    if (!res.ok) return;
    speciesRulesData = await res.json();
    renderSpeciesRules(document.getElementById('species-num').value);
  } catch (err) { console.warn('species rules load failed', err); }
}
function renderSpeciesRules(speciesNum) {
  const data = speciesRulesData[speciesNum];
  if (!data) return;
  document.getElementById('species-rules-title').textContent = data.label + ' のルール';
  document.getElementById('species-rules-summary').textContent = data.summary || '';
  const body = document.getElementById('species-rules-body');
  body.innerHTML = '';
  (data.sections || []).forEach((sec) => {
    const h = document.createElement('div');
    h.textContent = sec.heading;
    h.style.fontWeight = '600';
    h.style.color = '#1e293b';
    h.style.marginTop = '0.5em';
    body.appendChild(h);
    const ul = document.createElement('ul');
    ul.style.margin = '0.2em 0 0.4em 1.2em';
    ul.style.padding = '0';
    (sec.items || []).forEach((it) => {
      const li = document.createElement('li');
      li.textContent = it;
      li.style.margin = '0.1em 0';
      ul.appendChild(li);
    });
    body.appendChild(ul);
  });
}
document.getElementById('species-num').addEventListener('change', (e) => {
  renderSpeciesRules(e.target.value);
});
loadSpeciesRules();

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

document.getElementById('cf-download').addEventListener('click', async () => {
  const name = document.getElementById('cf-download').dataset.cfName;
  if (!name) return;
  try {
    const res = await fetch('/cantus-firmus/' + encodeURIComponent(name) + '.mid');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name + '.mid';
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert('MIDI のダウンロードに失敗: ' + err.message);
  }
});

// ---------- OSMD renderer ----------
// Note annotations(scale_degree + melodic interval)はサーバー側で
// MusicXML の <lyric> として埋め込まれて来るので、OSMD が自動的に音符
// の下に描いてくれる。クライアント側で SVG を弄る必要はない。
let osmd = null;
async function renderMusicXML(xml, issues) {
  const host = document.getElementById('score-host');
  host.innerHTML = '';
  if (!xml) return;
  if (!osmd || osmd._host !== host) {
    osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(host, {
      autoResize: true, drawTitle: false, drawSubtitle: false, drawComposer: false,
      drawLyrics: true,
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

// ---------- score playback (WebAudio from note_annotations) ----------
let scorePlayback = null;
let lastAnnotations = null;

function enableScorePlayback(annotations) {
  lastAnnotations = annotations;
  const btn = document.getElementById('score-play');
  btn.disabled = !(annotations && annotations.length);
}
function stopScorePlayback() {
  if (scorePlayback) {
    scorePlayback.forEach((n) => { try { n.stop(); } catch {} });
    scorePlayback = null;
  }
  document.getElementById('score-stop').style.display = 'none';
  document.getElementById('score-play').style.display = '';
}
function playScore() {
  if (!lastAnnotations || !lastAnnotations.length) return;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  stopScorePlayback();

  const beatDur = 0.5;  // seconds per quarter note (BPM 120)
  const now = audioCtx.currentTime + 0.05;
  const created = [];
  // Group by part for distinct timbres (slight detune).
  const parts = [...new Set(lastAnnotations.map((a) => a.part))];
  lastAnnotations.forEach((a) => {
    if (a.pitch == null || a.start_beat == null) return;
    const dur = (a.duration || 4.0) * beatDur;
    const start = now + (a.start_beat || 0) * beatDur;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    const partIdx = parts.indexOf(a.part);
    osc.type = partIdx === 0 ? 'triangle' : (partIdx === 1 ? 'sine' : 'square');
    osc.frequency.value = midiToFreq(a.pitch);
    osc.connect(gain).connect(audioCtx.destination);
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.12, start + 0.02);
    gain.gain.setValueAtTime(0.12, start + dur * 0.85);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    osc.start(start);
    osc.stop(start + dur + 0.05);
    created.push(osc);
  });
  scorePlayback = created;
  document.getElementById('score-stop').style.display = '';
  document.getElementById('score-play').style.display = 'none';
  const maxEnd = Math.max(...lastAnnotations.map((a) => ((a.start_beat || 0) + (a.duration || 4)) * beatDur));
  setTimeout(stopScorePlayback, (maxEnd + 1) * 1000);
}
document.getElementById('score-play').addEventListener('click', playScore);
document.getElementById('score-stop').addEventListener('click', stopScorePlayback);

// ---------- progress timer ----------
let _progressTimer = null;
function startProgress(label) {
  const out = document.getElementById('output');
  const t0 = Date.now();
  out.textContent = label + ' (0秒)';
  if (_progressTimer) clearInterval(_progressTimer);
  _progressTimer = setInterval(() => {
    const sec = Math.floor((Date.now() - t0) / 1000);
    if (out.textContent.startsWith(label)) out.textContent = label + ' (' + sec + '秒)';
    else clearInterval(_progressTimer);
  }, 1000);
}
function stopProgress() {
  if (_progressTimer) { clearInterval(_progressTimer); _progressTimer = null; }
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
  startProgress('送信中');
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
      enableScorePlayback(data.note_annotations);

      const fdXml = new FormData();
      for (const f of files) fdXml.append('files', f);
      fdXml.append('key', form.elements['key'].value || '');
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
      tutor.innerHTML = (typeof marked !== 'undefined' && marked.parse) ? marked.parse(data.critique) : data.critique;
      return;
    }
  } catch (err) {
    document.getElementById('output').textContent = 'エラー: ' + err.message;
  } finally {
    stopProgress(); setBusy(form, false);
  }
});

// ---------- learning tracks tab ----------
let tracksData = null;
async function loadTracks() {
  try {
    const res = await fetch('/tracks');
    if (!res.ok) return;
    tracksData = await res.json();
    const trackSel = document.getElementById('lesson-track');
    trackSel.innerHTML = '';
    tracksData.tracks.forEach((t) => {
      const opt = document.createElement('option');
      opt.value = t.id;
      opt.textContent = t.title;
      trackSel.appendChild(opt);
    });
    trackSel.dispatchEvent(new Event('change'));
  } catch (err) { console.warn('tracks load failed', err); }
}
loadTracks();

document.getElementById('lesson-track').addEventListener('change', (e) => {
  const track = tracksData?.tracks?.find((t) => t.id === e.target.value);
  if (!track) return;
  const lessonSel = document.getElementById('lesson-pick');
  lessonSel.innerHTML = '';
  track.lessons.forEach((l) => {
    const opt = document.createElement('option');
    opt.value = l.id;
    opt.textContent = l.title;
    lessonSel.appendChild(opt);
  });
  lessonSel.dispatchEvent(new Event('change'));
});

document.getElementById('lesson-pick').addEventListener('change', async (e) => {
  const trackId = document.getElementById('lesson-track').value;
  const lessonId = e.target.value;
  if (!trackId || !lessonId) return;
  try {
    const res = await fetch('/tracks/' + encodeURIComponent(trackId) + '/lessons/' + encodeURIComponent(lessonId));
    if (!res.ok) return;
    const lesson = await res.json();

    document.getElementById('lesson-title-display').textContent = lesson.title;
    document.getElementById('lesson-summary-display').textContent = lesson.summary || '';
    document.getElementById('lesson-intent-display').textContent = lesson.intent || '';

    const card = document.getElementById('lesson-rule-card');
    card.innerHTML = '';
    (lesson.rule_card || []).forEach((sec) => {
      const h = document.createElement('div');
      h.textContent = sec.heading;
      h.style.fontWeight = '600';
      h.style.color = '#1e293b';
      h.style.marginTop = '0.4em';
      card.appendChild(h);
      const ul = document.createElement('ul');
      ul.style.margin = '0.2em 0 0 1.2em';
      (sec.items || []).forEach((it) => {
        const li = document.createElement('li');
        li.textContent = it;
        ul.appendChild(li);
      });
      card.appendChild(ul);
    });

    const refsHost = document.getElementById('lesson-references');
    if (lesson.references && lesson.references.length) {
      refsHost.innerHTML = '参考: ' + lesson.references.map((r) => '<span style="margin-right:0.6em;">' + r + '</span>').join('');
    } else {
      refsHost.innerHTML = '';
    }

    // Cantus firmus row だけは 2 声レッスンの時に表示
    const isTwoPart = (lesson.expected_parts || []).includes('cantus_firmus');
    document.getElementById('lesson-cf-row').style.display = isTwoPart ? '' : 'none';
    document.getElementById('lesson-cf-dropzone').style.display = isTwoPart && !document.getElementById('lesson-preset').value ? '' : 'none';

    // Cantus preset 一覧をレッスン推奨で絞る
    const presetSel = document.getElementById('lesson-preset');
    presetSel.innerHTML = '';
    if (isTwoPart) {
      const empty = document.createElement('option');
      empty.value = '';
      empty.textContent = '(アップロードする)';
      presetSel.appendChild(empty);
      (lesson.cantus_firmus_presets || []).forEach((name) => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        presetSel.appendChild(opt);
      });
    }

    // Heading の文言だけ調整(複数声部レッスンでは "Counterpoint" 表記を消す)
    document.getElementById('lesson-cp-heading').textContent = isTwoPart ? 'Counterpoint(あなたの解答)' : '提出する MIDI(全声部を1ファイルに)';
  } catch (err) { console.warn('lesson load failed', err); }
});

document.getElementById('lesson-preset').addEventListener('change', (e) => {
  const isUpload = !e.target.value;
  document.getElementById('lesson-cf-dropzone').style.display = isUpload ? '' : 'none';
});

document.getElementById('lesson-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const cpFiles = lessonCpDz.files();
  if (!cpFiles.length) { alert('提出する MIDI を選んでください'); return; }

  const trackId = form.elements['track_id'].value;
  const lessonId = form.elements['lesson_id'].value;
  const presetVal = form.elements['preset'].value;
  const cfFiles = lessonCfDz.files();
  const isCfRowVisible = document.getElementById('lesson-cf-row').style.display !== 'none';

  if (isCfRowVisible && !presetVal && !cfFiles.length) {
    alert('Cantus firmus を選ぶか、プリセットを選んでください');
    return;
  }

  const fd = new FormData();
  fd.append('track_id', trackId);
  fd.append('lesson_id', lessonId);
  fd.append('counterpoint', cpFiles[0]);
  if (isCfRowVisible) {
    if (presetVal) fd.append('preset', presetVal);
    else if (cfFiles.length) fd.append('cantus_firmus', cfFiles[0]);
  }
  const key = form.elements['key'].value;
  if (key) fd.append('key', key);
  if (e.submitter.dataset.mode === 'check-llm') fd.append('use_llm', 'true');

  setBusy(form, true);
  startProgress('送信中(レッスン)');
  document.getElementById('tutor').style.display = 'none';
  document.getElementById('tutor-heading').style.display = 'none';

  try {
    const data = await postJSON('/lesson', fd);
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    const result = data.result;
    renderIssues(result.issues);
    enableScorePlayback(data.note_annotations);
    if (data.musicxml) await renderMusicXML(data.musicxml, result.issues);
    if (data.tutor_feedback) {
      document.getElementById('tutor').style.display = '';
      document.getElementById('tutor-heading').style.display = '';
      document.getElementById('tutor').innerHTML = (typeof marked !== 'undefined' && marked.parse) ? marked.parse(data.tutor_feedback) : data.tutor_feedback;
    }
  } catch (err) {
    document.getElementById('output').textContent = 'エラー: ' + err.message;
  } finally {
    stopProgress(); setBusy(form, false);
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
  startProgress('送信中(Species)');
  document.getElementById('tutor').style.display = 'none';
  document.getElementById('tutor-heading').style.display = 'none';

  try {
    const data = await postJSON('/' + mode, fd);
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    const result = data.result;
    renderIssues(result.issues);
    enableScorePlayback(data.note_annotations);
    if (data.musicxml) await renderMusicXML(data.musicxml, result.issues);
    if (data.tutor_feedback) {
      document.getElementById('tutor').style.display = '';
      document.getElementById('tutor-heading').style.display = '';
      document.getElementById('tutor').innerHTML = (typeof marked !== 'undefined' && marked.parse) ? marked.parse(data.tutor_feedback) : data.tutor_feedback;
    }
  } catch (err) {
    document.getElementById('output').textContent = 'エラー: ' + err.message;
  } finally {
    stopProgress(); setBusy(form, false);
  }
});
</script>
</body>
</html>
"""
