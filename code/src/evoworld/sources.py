"""Adapters for local source statistics; trajectories are never copied."""
from __future__ import annotations
import csv, json
from pathlib import Path

def casas_histogram(path):
    counts={}
    with Path(path).open() as f:
        for row in csv.reader(f):
            if len(row)>=3 and row[-1].strip().upper()=="ON": counts[row[2].strip().casefold()]=counts.get(row[2].strip().casefold(),0)+1
    return {"source":"CASAS_Aruba","room_counts":counts,"trajectory_copied":False}

def local_source_manifest(root):
    root=Path(root); return {"CASAS":(root/"casas").exists(),"HD_EPIC":(root/"hd_epic_annotations").exists(),"ParaHome":(root/"parahome_data").exists(),"HOMER_PLUS":(root/"HOMER_PLUS").exists()}
