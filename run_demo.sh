#!/bin/bash

echo "======================================"
echo "FedSymGraph Demo"
echo "======================================"
echo ""
echo "This demo runs a federated learning simulation with:"
echo "  - 1 Global Coordinator (Server)"
echo "  - 2 Federated Clients (Domain Nodes)"
echo ""
echo "Press Ctrl+C to stop all processes"
echo ""

trap 'kill $(jobs -p) 2>/dev/null' EXIT

echo "[1/3] Starting Global Coordinator..."
python server.py --rounds 3 --min-clients 2 --port 8080 &
SERVER_PID=$!
sleep 3

echo "[2/3] Starting Client 1 (HQ Domain)..."
python client.py --client_id 1 --server 127.0.0.1:8080 --no-llm &
CLIENT1_PID=$!
sleep 2

echo "[3/3] Starting Client 2 (IoT Branch Domain)..."
python client.py --client_id 2 --server 127.0.0.1:8080 --no-llm &
CLIENT2_PID=$!

wait
