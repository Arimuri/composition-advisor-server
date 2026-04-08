# composition-advisor-server

[composition_advisor](https://github.com/Arimuri/ITBS_TAMANEGI-SENSEI/tree/main/composition_advisor) を HTTP で叩けるようにする FastAPI ラッパー。MIDI ファイルを投げると、ブラウザで **譜面表示 + 問題箇所のハイライト + Claude による添削** が返ってきます。

有村の音楽学習用ツール置き場です。

## できること

### 一般分析
- 複数 MIDI ファイル(melody / chord / bass …)を投げて、コード進行・度数・声部進行を分析
- 半音衝突 / 声部交叉 / 平行5度8度 / 隠伏進行 / 音域逸脱 / コードトーン外 を検出
- Claude にそのまま投げてジャズ/フュージョン文脈の自然言語添削
- 検出ルールに対応した修正 MIDI(`*_fixed.mid`)+ diff レポートを zip で出力

### 対位法レッスン(Species Counterpoint)
- Fux 流の **Species 1〜5**(1:1 / 2:1 / 4:1 / 掛留 / florid)に対応
- 内蔵 cantus firmus プリセット 6 種(D dorian / E phrygian / F lydian / G mixolydian / A aeolian / C major)
- プリセットは **譜面プレビュー / WebAudio で簡易再生 / MIDI ファイルとして DL** が可能
- 自分で書いた counterpoint を投げると Species ごとのルールで採点 + 譜面に問題箇所をハイライト
- 「対位法教師」モードで Claude が Fux/Jeppesen 風に添削

## エンドポイント

| メソッド | パス | 用途 |
|---|---|---|
| GET | `/healthz` | ヘルスチェック(認証不要) |
| GET | `/` | アップロード用 HTML(ドラッグ&ドロップ + タブ UI) |
| POST | `/analyze` | MIDI を multipart で投げる → AnalysisResult JSON |
| POST | `/critique` | 上記 + Claude の添削テキスト |
| POST | `/fix` | 修正 MIDI(各パート)+ diff レポートの zip |
| POST | `/musicxml` | MIDI → MusicXML(ブラウザ譜面描画用) |
| POST | `/species` | counterpoint + cantus firmus → species 検査結果 + MusicXML |
| POST | `/species-tutor` | 同上 + Claude 教師の添削 |
| GET | `/species-presets` | 内蔵 cantus firmus プリセット一覧 |
| GET | `/cantus-firmus/{name}.mid` | プリセットを SMF として配信(DL 用) |
| GET | `/cantus-firmus/{name}.musicxml` | プリセットを MusicXML として配信 |

## ローカル起動

```bash
uv sync
uv run uvicorn composition_advisor_server.app:app --reload --port 8765
# → http://127.0.0.1:8765/
```

依存している [composition_advisor](https://github.com/Arimuri/ITBS_TAMANEGI-SENSEI/tree/main/composition_advisor) は GitHub 経由で `uv sync` 時に自動取得されます。

## 環境変数

| 変数 | デフォルト | 用途 |
|---|---|---|
| `COMPOSITION_ADVISOR_USER` | (なし) | HTTP Basic 認証ユーザー名 |
| `COMPOSITION_ADVISOR_PASSWORD` | (なし) | HTTP Basic 認証パスワード |
| `ANTHROPIC_API_KEY` | (なし) | Claude API キー(`/critique` / `/species-tutor` / `/fix?use_llm=true` で必須) |
| `COMPOSITION_ADVISOR_CONCURRENCY` | `1` | 重い処理(chordify / species)の同時実行数 |
| `COMPOSITION_ADVISOR_MAX_UPLOAD` | `2097152` (2 MB) | 1 ファイルあたりのアップロード上限バイト数 |

`COMPOSITION_ADVISOR_USER` が未設定なら全エンドポイント認証なし(ローカル開発用)。LLM 系エンドポイントは `ANTHROPIC_API_KEY` 未設定だと 503 を返します。

## 使用例(curl)

```bash
# ヘルスチェック
curl http://127.0.0.1:8765/healthz

# 一般分析
curl -F "files=@melody.mid" -F "files=@chord.mid" -F "key=C" \
     http://127.0.0.1:8765/analyze | jq

# Claude 添削
curl -u arimura:secret \
     -F "files=@melody.mid" -F "files=@chord.mid" -F "key=C" \
     http://127.0.0.1:8765/critique | jq

# 修正 MIDI 出力(zip)
curl -u arimura:secret \
     -F "files=@melody.mid" -F "files=@chord.mid" -F "key=C" \
     -o fix_output.zip \
     http://127.0.0.1:8765/fix

# 対位法 Species 1 検査(プリセット使用)
curl -u arimura:secret \
     -F "counterpoint=@my_cp.mid" \
     -F "preset=c_major_short" \
     -F "species_num=1" \
     http://127.0.0.1:8765/species | jq

# Cantus firmus プリセットを MIDI として DL
curl -u arimura:secret \
     -o c_major_short.mid \
     http://127.0.0.1:8765/cantus-firmus/c_major_short.mid
```

## VPS デプロイ(自分用メモ)

さくら VPS 1GB プランを想定。

```bash
# 1. リポジトリを clone
ssh ubuntu@<vps>
cd ~/projects
git clone https://github.com/Arimuri/composition-advisor-server.git
cd composition-advisor-server
uv sync

# 2. 認証情報を root 所有で保存
sudo mkdir -p /etc/composition-advisor-server
sudo nano /etc/composition-advisor-server/env
# COMPOSITION_ADVISOR_USER=...
# COMPOSITION_ADVISOR_PASSWORD=...
# ANTHROPIC_API_KEY=sk-ant-...
sudo chmod 600 /etc/composition-advisor-server/env

# 3. systemd ユニット例
sudo tee /etc/systemd/system/composition-advisor-server.service > /dev/null <<'EOF'
[Unit]
Description=composition-advisor-server (FastAPI)
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/projects/composition-advisor-server
EnvironmentFile=/etc/composition-advisor-server/env
ExecStart=/home/ubuntu/.local/bin/uv run uvicorn composition_advisor_server.app:app --host 127.0.0.1 --port 8765
Restart=on-failure

# 1GB プランでは KYOTEI_AI 等と共存するためメモリを絞る
MemoryMax=400M
MemoryHigh=300M
CPUQuota=80%

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ReadWritePaths=/home/ubuntu/projects/composition-advisor-server

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now composition-advisor-server
```

公開する場合は caddy + sslip.io(または独自ドメイン)で reverse proxy + Let's Encrypt が一番楽です。

```Caddyfile
{
  email you@example.com
}

49-212-203-123.sslip.io {
  reverse_proxy 127.0.0.1:8765
  encode gzip
}
```

ufw で 22 / 80 / 443 を許可、さくら VPS 側のパケットフィルタも同様に開放してください。

## 設計メモ

- **State なし**: アップロードファイルは処理が終わったら捨てます(キャッシュ無し)。再現には毎回 MIDI を投げ直してください。
- **同時実行 1 リクエスト**: chordify / species 系は CPU/メモリを食うので、`asyncio.Semaphore` で 1 個ずつ処理します。`COMPOSITION_ADVISOR_CONCURRENCY` で増やせます。
- **音名表記**: ブラウザ・LLM への表示はすべて Studio One 表記(中央 C = C3)に統一しています。
- **譜面描画**: ブラウザ側で [OpenSheetMusicDisplay](https://opensheetmusicdisplay.org/) を CDN から読み込んで MusicXML を描画。サーバーは画像を返しません。

## ライセンス

特に明記なし(個人ツール置き場)。フォークして使う場合は自己責任でどうぞ。
