#!/usr/bin/env bash
# Astra を最後まで回す．クレジット切れ（run_astra_clips.py の終了コード 75）や認証切れで止まったら，
# 一定間隔で再試行する．失敗した試行は開始 3 秒ほどで終わり，費用は掛からない．成功済みのバッチは飛ばされる．
# 全バッチが成功したら終了する．止めるときは STOP ファイルを作る（touch <frames-root>/STOP）か，プロセスを kill する．
#
#   bash resume_astra.sh <frames-root> <prompt-file> <team-a-colour> <team-b-colour> [interval_s]
# 色なし版プロンプトは TEAM_A_NAME=T1 TEAM_B_NAME=T2 を付けて呼ぶ（色は使われないので "unused" を渡す）．
# 例: TEAM_A_NAME=T1 TEAM_B_NAME=T2 bash resume_astra.sh work/IMG_0114 prompts/rendered_nocolor/IMG_0114.txt unused unused
set -u
root=$1; prompt=$2; ca=$3; cb=$4; interval=${5:-3600}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$HERE/run_astra_clips.py"
log="$root/resume.log"

remaining() {  # 成功していないバッチの数
  python3 -I - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
want = {p.parent.name for p in root.glob("*/manifest.json")}
done = set()
res = root / "results.jsonl"
if res.exists():
    for line in res.read_text().splitlines():
        r = json.loads(line)
        if r.get("returncode") == 0 and r.get("parsed") and not r.get("schema_error"):
            done.add(r["clip"])
print(len(want - done))
PY
}

while true; do
  [ -e "$root/STOP" ] && { echo "$(date +%FT%T) STOP file: exit" >> "$log"; exit 0; }
  n=$(remaining)
  [ "$n" = 0 ] && { echo "$(date +%FT%T) all batches done" >> "$log"; exit 0; }
  echo "$(date +%FT%T) start: $n batches remaining" >> "$log"
  python3 "$runner" --frames-root "$root" --prompt-file "$prompt" --team-a "${TEAM_A_NAME:-A}" "$ca" --team-b "${TEAM_B_NAME:-B}" "$cb" >> "$root/run.log" 2>&1
  rc=$?
  echo "$(date +%FT%T) runner exit $rc, $(remaining) batches remaining" >> "$log"
  [ "$(remaining)" = 0 ] && { echo "$(date +%FT%T) all batches done" >> "$log"; exit 0; }
  sleep "$interval"
done
