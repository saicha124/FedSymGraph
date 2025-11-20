#!/bin/bash

export LD_LIBRARY_PATH="/nix/store/bmi5znnqk4kg2grkrhk6py0irc8phf6l-gcc-14.2.1.20250322-lib/lib:$LD_LIBRARY_PATH"
PYTHON=".pythonlibs/bin/python"

echo "======================================"
echo "Testing Differential Privacy"
echo "======================================"
echo ""
echo "Starting server and clients WITH privacy enabled"
echo ""

trap 'kill $(jobs -p) 2>/dev/null' EXIT

echo "[1/3] Starting Global Coordinator..."
$PYTHON server.py --rounds 2 --min-clients 2 --port 8080 &
SERVER_PID=$!
sleep 3

echo "[2/3] Starting Client 1 WITH PRIVACY..."
$PYTHON client.py --client_id 1 --server 127.0.0.1:8080 --enable-privacy --no-llm &
CLIENT1_PID=$!
sleep 2

echo "[3/3] Starting Client 2 WITH PRIVACY..."
$PYTHON client.py --client_id 2 --server 127.0.0.1:8080 --enable-privacy --no-llm &
CLIENT2_PID=$!

wait
