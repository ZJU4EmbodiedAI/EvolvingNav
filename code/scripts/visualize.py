#!/usr/bin/env python3
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def timeline(out):
    fig, ax = plt.subplots(figsize=(10, 2.8)); ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    for x, label, color in [(0.4, "Past RGB-D\nobservations", "#4b5563"), (4.0, "Hidden\nevolution", "#f4a261"), (7.1, "Query +\nreplanning", "#16a9b8")]:
        ax.add_patch(FancyBboxPatch((x, .28), 2.0, .42, boxstyle="round,pad=.03", facecolor=color, alpha=.92, edgecolor="none")); ax.text(x+1, .49, label, color="white", ha="center", va="center", fontsize=12, weight="bold")
    ax.annotate("", (3.8,.49), (2.55,.49), arrowprops={"arrowstyle":"->", "lw":2, "color":"#374151"}); ax.annotate("", (6.9,.49), (5.95,.49), arrowprops={"arrowstyle":"->", "lw":2, "color":"#374151"}); fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)

def beliefs(out):
    names=["Kitchen cabinet", "Dining table", "Unknown"]; values=[.65,.20,.15]; fig, ax=plt.subplots(figsize=(7,2.7)); ax.barh(names[::-1], values[::-1], color=["#f4a261","#16a9b8","#4b5563"]); ax.set_xlim(0,1); ax.set_xlabel("Posterior probability"); ax.spines[["top","right","left"]].set_visible(False)
    for i,v in enumerate(values[::-1]): ax.text(v+.02,i,f"{v:.0%}",va="center",weight="bold")
    fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)

def trajectory(out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    last_seen = [(0.4, 0.8), (2.0, 0.8), (3.6, 0.8), (5.2, 0.8)]
    predictive = [(0.4, 0.2), (2.0, 0.35), (3.6, 0.2), (5.2, 0.55), (6.5, 0.75)]
    ax.plot(*zip(*last_seen), color="#4b5563", lw=2.5, marker="o", label="Last-seen route")
    ax.plot(*zip(*predictive), color="#16a9b8", lw=2.5, marker="o", label="Predictive route")
    ax.scatter([3.6], [0.2], color="#f4a261", s=100, zorder=3, label="Hidden relocation")
    ax.set(xlim=(0, 7), ylim=(0, 1), xlabel="Navigation progress", ylabel="Candidate-space projection")
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)

parser=argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); a=parser.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); timeline(a.output_dir/"timeline.png"); beliefs(a.output_dir/"belief.png"); trajectory(a.output_dir/"trajectory.png")
