# EvolvingNav

## Beyond the Remembered World

**Predictive 4D Belief for Persistent Navigation in Evolving Worlds**

EvolvingNav is an embodied navigation framework for settings where an agent's remembered scene can become stale while it is away or moving. It forecasts whether an observed state will persist, reasons about plausible relocation, and uses visibility-aware RGB-D evidence to revise its belief and replan.

**Mingjian Gao**<sup>1,*</sup>, **Zhaocheng Li**<sup>1,*</sup>, **Haoyang Huang**<sup>2,*</sup>, **Wenqiao Zhang**<sup>1,‡,†</sup>, **Yingjie NIU**<sup>3,4,‡,†</sup>, **Hao Zhou**<sup>1</sup>, **Chao LI**<sup>3</sup>, **Juncheng LI**<sup>1</sup>, **Siliang Tang**<sup>1</sup>, **Yueting Zhuang**<sup>1</sup>

<sup>1</sup> Zhejiang University · <sup>2</sup> University of California, San Diego · <sup>3</sup> Deeprobotics · <sup>4</sup> The Chinese University of Hong Kong<br />
<sup>*</sup> Equal contribution · <sup>‡</sup> Corresponding authors · <sup>†</sup> Project lead

## At a glance

| 54 scenes | 803.68K tasks | 61.32% first-inspection SR | 86.18% search SR |
|:--:|:--:|:--:|:--:|
| EvoWorld-Bench | executable instances | stale-memory decisions | budgeted search |

## Overview

<p align="center"><img src="figures/overview_web_latest.png" alt="EvolvingNav overview" width="960" /></p>

EvolvingNav turns timestamped 3D histories into an arrival-time belief over the current world. The agent predicts, acts for a short horizon, checks whether newly visible evidence should have been observable, and updates its memory before continuing.

## Method

<p align="center"><img src="figures/method_web_latest.png" alt="EvolvingNav method" width="960" /></p>

The framework combines memory construction, state prediction, visibility-aware evidence filtering, and belief-guided route planning. Candidate states are forecast at their estimated arrival times instead of only at the instant of the query.

## EvoWorld-Bench

<p align="center"><img src="figures/benchmark_figure4_web.png" alt="EvoWorld-Bench construction" width="620" /> <img src="figures/benchmark_figure3_web.png" alt="EvoWorld-Bench protocols" width="620" /></p>

The benchmark grounds human traces into executable evolving worlds and evaluates predictive navigation, belief-guided search, evidence-aware replanning, online dynamics, held-out transfer, and embodied question answering.

## Results

| Method | FindingDory HL-SR | GOAT-Bench SR | First-Inspection SR | Search SR | SPL |
|:--|--:|--:|--:|--:|--:|
| DynaMem | 30.30 | 14.54 | 45.33 | 71.94 | 56.83 |
| **EvolvingNav (ours)** | **53.22 ± 3.87** | **35.43 ± 2.16** | **61.32 ± 3.24** | **86.18 ± 2.07** | **70.15 ± 1.74** |

<p align="center"><img src="figures/case_study_web_latest.png" alt="Qualitative cases" width="960" /></p>

## Real-world validation

Across 64 matched LYNX M20 search episodes, EvolvingNav reaches 34.4% first-inspection success, 48.4% search success, and 24.3% recovery success, with 43.8 m mean travel.

## Code

Runnable implementation, Habitat runners, training scripts, and tests are in [`code/`](code/README.md).

```bash
cd code
python -m pip install -r requirements.txt
export PYTHONPATH=.:src:scripts
python -m pytest tests -q
```

For dataset paths, model training, and benchmark commands, see [`code/README.md`](code/README.md).

## Project page

The complete paper website is available at [zju4embodiedai.github.io/EvolvingNav](https://zju4embodiedai.github.io/EvolvingNav/).

## Citation

```bibtex
@inproceedings{evolvingnav2027,
  title     = {Beyond the Remembered World: Predictive 4D Belief for Persistent Navigation in Evolving Worlds},
  author    = {Gao, Mingjian and Li, Zhaocheng and Huang, Haoyang and Zhang, Wenqiao and Niu, Yingjie and Zhou, Hao and Li, Chao and Li, Juncheng and Tang, Siliang and Zhuang, Yueting},
  year      = {2026}
}
```
