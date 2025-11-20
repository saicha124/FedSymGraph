#!/bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/nix/store/$(ls /nix/store | grep 'gcc.*lib$' | head -1)/lib"
python server.py --rounds 3 --min-clients 2
