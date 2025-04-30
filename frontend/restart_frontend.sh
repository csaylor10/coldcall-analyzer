#!/bin/bash
cd /workspace/fastapi_project/frontend
apt update && apt install -y lsof
PIDS=$(lsof -t -i :3000)
if [ ! -z "$PIDS" ]; then
  echo "Killing existing processes on port 3000: $PIDS"
  kill -9 $PIDS
  sleep 3
fi
npm start
