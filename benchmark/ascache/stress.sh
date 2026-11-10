#!/usr/bin/env bash
ascache_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${ascache_repo_root}/bench_ascache.py" --host 127.0.0.1 --port 8000 --tokenizer /DeepSeek-V4.1-Flash-20260910 --total-len 61440 --page-size 256 --target-hit-rates "0.90" --qps 500 --duration 1 --connection-limit 256 --max-concurrency 256 --num-requests "${BENCH_REQUESTS:-480}" --seed "${BENCH_SEED:-20261022}" --json-backend "${BENCH_JSON_BACKEND:-orjson}" "$@"
