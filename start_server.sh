#!/bin/bash
export LD_LIBRARY_PATH="/nix/store/bmi5znnqk4kg2grkrhk6py0irc8phf6l-gcc-14.2.1.20250322-lib/lib:$LD_LIBRARY_PATH"
python server.py --rounds 3 --min-clients 2
