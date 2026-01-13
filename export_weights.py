import os
import torch
from models.cage4 import load  # eure load()-Funktion

contractor_active = True

if contractor_active:
    SRC = "checkpoints/contractoractive"   # <-- fname aus train.py
    DST = "weights/contractor_active"
else:
    SRC = "checkpoints/contractorinactive"   # <-- fname aus train.py
    DST = "weights/contractor_inactive"

os.makedirs(DST, exist_ok=True)

for i in range(5):
    src = f"{SRC}-{i}_checkpoint.pt"
    dst = f"{DST}/gnn_ppo-{i}.pt"

    ckpt = torch.load(src, map_location="cpu")

    # optional: sanity check
    assert "actor" in ckpt and "critic" in ckpt and "agent" in ckpt

    torch.save(ckpt, dst)
    print(f"Exported {dst}")
