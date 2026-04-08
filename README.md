# composition-advisor-server

[composition_advisor](https://github.com/Arimuri/ITBS_TAMANEGI-SENSEI/tree/main/composition_advisor) を HTTP で叩けるようにする FastAPI ラッパー。

ブラウザから MIDI を投げて分析・添削・修正MIDIを受け取れる薄いサーバー。

## エンドポイント

| メソッド | パス | 用途 |
|---|---|---|
| GET | `/` | アップロード用HTML(ブラウザ用) |
| GET | `/healthz` | ヘルスチェック(認証不要) |
| POST | `/analyze` | MIDIをmultipart投げる → AnalysisResult JSON |
| POST | `/critique` | 上記 + Claudeの添削テキスト返却 |
| POST | `/fix` | 修正済みMIDI + diffレポート(zip) |

## ローカル起動

```bash
uv sync
uv run uvicorn composition_advisor_server.app:app --reload --port 8765
# → http://127.0.0.1:8765/
```

## 認証

環境変数で HTTP Basic 認証:

```bash
export COMPOSITION_ADVISOR_USER=arimura
export COMPOSITION_ADVISOR_PASSWORD='秘密'
```

未設定なら全エンドポイント認証なし(ローカル開発用)。

## Claude 添削を有効化

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

未設定だと `/critique` `/fix?use_llm=true` は 503 を返します。

## 使用例

```bash
# ヘルスチェック
curl http://127.0.0.1:8765/healthz

# 分析
curl -F "files=@melody.mid" -F "files=@chord.mid" -F "key=C" \
     http://127.0.0.1:8765/analyze | jq

# 添削
curl -u arimura:secret \
     -F "files=@melody.mid" -F "files=@chord.mid" -F "key=C" \
     http://127.0.0.1:8765/critique | jq

# 修正MIDI
curl -u arimura:secret \
     -F "files=@melody.mid" -F "files=@chord.mid" -F "key=C" \
     -o fix_output.zip \
     http://127.0.0.1:8765/fix
```

## VPS デプロイ

caddy + sslip.io + systemd で自動起動する想定。詳細は別途。
