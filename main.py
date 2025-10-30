import subprocess
import os
from time import sleep
import json
import select
import sys
"""
All options:
    max_eps 
    comment
    mask_enable
    enable_messages
    prio_weights
    enable_prio
    always_restore
    aggressive_analyse
    aggressive_analyse_rep
    enforce_connections

"""
"""
Base configuration:
    max_eps=100,
    comment=Testing
    mask_messages=True,
    prio_weights=[3, 2, 1],
    enable_prio=False,
    always_restore=False,
    aggressive_analyse=False,
    aggressive_analyse_rep=[3, 2],
    enforce_connections = True
"""
load_done = False
consecutiv_runs = 6
counter = 0
#fixed settings
max_eps = "100"
path_config = os.path.join(os.path.dirname(__file__), 'config.json')
path_done = os.path.join(os.path.dirname(__file__), 'done.json')
# Run the command to open a new terminal and execute the script.
# We use Popen for a non-blocking call.
with open(path_config,'r', encoding='utf-8') as f:
    configs = json.load(f)
if load_done:
    try:
        with open(path_done,'r', encoding='utf-8') as f:
            done = json.load(f)
    except FileNotFoundError:
        print("Done.json not found")
    assert len(done) == len(configs), "Done list length does not match configs length"
else:
    done = [0]*len(configs)

for idx, config in enumerate(configs):
    if done[idx] == 1:
        print(f"Configuration {idx} already done, skipping.")
        continue
    com = f"python3 evaluation.py --max-eps {max_eps}"    
    for key in config:
        print(f"{key}: {config[key]}")
        com += f" --{key} {config[key]}"
    print(f"Running command in a new terminal: {' '.join(com)}")
    try:
        subprocess.Popen(com, shell=True)
        sleep(1)
        subprocess.Popen(com, shell=True)
        sleep(1)
        subprocess.run(com, shell=True, check=True)
    except FileNotFoundError:
        print("--Failed--")
    sleep(1)  # Optional: wait a bit before opening the next terminal
    done[idx] = 1
    counter += 1
    print("Enter 1 to continue with testing or 0 to abort and save progress.\nThis will auto-abort in 300 seconds.")
    if counter % consecutiv_runs ==0:
        user_input, _, _  = select.select([sys.stdin], [], [], 300)
        if user_input:
            line = sys.stdin.readline ().strip()
            if line == '1':
                print("Continuing with testing.")
            else:
                print("Aborting testing and saving progress.")
                break
        else:
            print("Aborting testing and saving progress, because of no response in time.")
            break

with open(path_done, 'w', encoding='utf-8') as f:
    json.dump(done, f, indent=2, ensure_ascii=False)