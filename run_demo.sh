#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNTIME_DIR="$SCRIPT_DIR/.demo"
PID_FILE="$RUNTIME_DIR/server.pid"
LOG_FILE="$RUNTIME_DIR/server.log"
BACKEND_PID_FILE="$RUNTIME_DIR/backend.pid"
BACKEND_LOG_FILE="$RUNTIME_DIR/backend.log"
VENV_DIR="$RUNTIME_DIR/venv"
WHISPER_MODEL_NAME="${WHISPER_MODEL_NAME:-large-v3-turbo}"
WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH:-$RUNTIME_DIR/models/whisper-$WHISPER_MODEL_NAME}"
SERVER_PORT=5173
EXPECT_PORT_VALUE=false

for argument in "$@"; do
  if [[ "$EXPECT_PORT_VALUE" == true ]]; then
    SERVER_PORT="$argument"
    EXPECT_PORT_VALUE=false
    continue
  fi

  case "$argument" in
    --port)
      EXPECT_PORT_VALUE=true
      ;;
    --port=*)
      SERVER_PORT="${argument#--port=}"
      ;;
  esac
done

stop_process_tree() {
  local parent_pid="$1"
  local child_pid

  while IFS= read -r child_pid; do
    if [[ -n "$child_pid" ]]; then
      stop_process_tree "$child_pid"
    fi
  done < <(pgrep -P "$parent_pid" 2>/dev/null || true)

  kill "$parent_pid" 2>/dev/null || true
}

stop_port_processes() {
  local port="$1"
  local process_id

  if ! command -v lsof >/dev/null 2>&1; then
    return
  fi

  while IFS= read -r process_id; do
    if [[ "$process_id" =~ ^[0-9]+$ ]]; then
      echo "포트 ${port}에서 실행 중인 기존 프로세스를 종료합니다. PID: $process_id"
      stop_process_tree "$process_id"
    fi
  done < <(lsof -ti "tcp:$port" 2>/dev/null | sort -u || true)
}

wait_for_http() {
  local url="$1"
  local process_id="$2"
  local attempts="$3"

  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    if curl --silent --fail --max-time 1 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$process_id" 2>/dev/null; then
      return 1
    fi
    sleep 0.2
  done
  return 1
}

cleanup_started_processes() {
  local pid_file
  local process_id
  for pid_file in "$PID_FILE" "$BACKEND_PID_FILE"; do
    [[ -f "$pid_file" ]] || continue
    process_id="$(<"$pid_file")"
    if [[ "$process_id" =~ ^[0-9]+$ ]]; then
      stop_process_tree "$process_id"
    fi
    rm -f "$pid_file"
  done
}

if ! command -v node >/dev/null 2>&1; then
  echo "오류: Node.js가 설치되어 있지 않습니다."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "오류: npm이 설치되어 있지 않습니다."
  exit 1
fi

if [[ ! -f package.json ]]; then
  echo "오류: $SCRIPT_DIR/package.json 파일을 찾을 수 없습니다."
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "오류: 의존성이 설치되어 있지 않습니다. 먼저 'npm install'을 실행하세요."
  exit 1
fi

mkdir -p "$RUNTIME_DIR"

for managed_pid_file in "$PID_FILE" "$BACKEND_PID_FILE"; do
  if [[ ! -f "$managed_pid_file" ]]; then
    continue
  fi
  EXISTING_PID="$(<"$managed_pid_file")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "기존 EasyListner 프로세스를 종료합니다. PID: $EXISTING_PID"
    stop_process_tree "$EXISTING_PID"

    for _attempt in {1..30}; do
      if ! kill -0 "$EXISTING_PID" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done

    if kill -0 "$EXISTING_PID" 2>/dev/null; then
      echo "기존 서버를 강제로 종료합니다. PID: $EXISTING_PID"
      kill -9 "$EXISTING_PID" 2>/dev/null || true
    fi
  fi
  rm -f "$managed_pid_file"
done

stop_port_processes "$SERVER_PORT"
stop_port_processes 8000

echo "EasyListner 데모 서버를 시작합니다."

if [[ ! -f "$WHISPER_MODEL_PATH/config.json" || ! -f "$WHISPER_MODEL_PATH/weights.safetensors" ]]; then
  echo "오류: MLX Whisper 모델이 없습니다. 먼저 ./setup_demo.sh을 실행하세요."
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/uvicorn" ]]; then
  echo "오류: STT 백엔드가 없습니다. 먼저 ./setup_demo.sh을 실행하세요."
  exit 1
fi

nohup env \
  WHISPER_MODEL_NAME="$WHISPER_MODEL_NAME" \
  WHISPER_MODEL_PATH="$WHISPER_MODEL_PATH" \
  "$VENV_DIR/bin/uvicorn" app:app \
  --app-dir "$SCRIPT_DIR/backend" \
  --host 127.0.0.1 \
  --port 8000 \
  >"$BACKEND_LOG_FILE" 2>&1 </dev/null &
BACKEND_PID=$!
echo "$BACKEND_PID" >"$BACKEND_PID_FILE"

if ! wait_for_http "http://127.0.0.1:8000/health" "$BACKEND_PID" 75; then
  cleanup_started_processes
  echo "오류: STT 백엔드를 시작하지 못했습니다."
  echo "로그를 확인하세요: $BACKEND_LOG_FILE"
  exit 1
fi

nohup env EASYLISTENER_LOCAL_DEMO=1 npm run dev -- "$@" >"$LOG_FILE" 2>&1 </dev/null &
SERVER_PID=$!
echo "$SERVER_PID" >"$PID_FILE"

if ! wait_for_http "http://localhost:$SERVER_PORT/" "$SERVER_PID" 100; then
  cleanup_started_processes
  echo "오류: 데모 서버를 시작하지 못했습니다."
  echo "로그를 확인하세요: $LOG_FILE"
  exit 1
fi

echo "백그라운드 실행이 완료되었습니다."
echo "Vite PID: $SERVER_PID · API/MLX PID: $BACKEND_PID"
echo "접속 주소: http://localhost:$SERVER_PORT/"
echo "로그 확인: tail -f \"$LOG_FILE\""
echo "백엔드 로그: tail -f \"$BACKEND_LOG_FILE\""
