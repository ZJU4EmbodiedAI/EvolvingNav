# EvolvingNav Agent

## Code

| Path | Function |
| --- | --- |
| `src/readyagent/p4d_belief/` | Continuous-time history encoder and persistence–relocation belief |
| `evolvingnav_paper/memory.py` | Causal entity versions, RGB-D backprojection and evidence provenance |
| `evolvingnav_paper/transition_model.py` | Row-normalized chronological transition head |
| `evolvingnav_paper/filter.py` | Current-time belief, arrival forecasts and evidence rounds |
| `evolvingnav_paper/coverage.py`, `calibration.py` | Online depth coverage and validation-fitted detection probability |
| `evolvingnav_paper/agent.py`, `controller.py` | Event-driven actions and frozen VLM tool selection |
| `evolvingnav_paper/world.py`, `backend.py`, `run.py` | Habitat action adapter and benchmark runner |
| `scripts/` | Dataset packing, belief/transition training and calibration |
| `tests/` | Unit tests |

## Environment

Use Python 3.11 with Habitat-Sim 0.3.3 installed.

```bash
cd code
python -m pip install -r requirements.txt
export PYTHONPATH=.:src:scripts
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export P4D_DATASET=/absolute/path/to/training_dataset
export NAV_TASKS=/absolute/path/to/navigation_tasks
export HSSD_ROOT=/absolute/path/to/hssd-hab
export NAVMESH_ROOT=/absolute/path/to/hssd-hab/navmeshes
export OPENAI_API_KEY="<your-api-key>"
```

## Train

If `records/packed/{train,val,test}.npz` are absent:

```bash
python scripts/pack_p4d_hssd_records.py --root "$P4D_DATASET"
```

Train the query-time belief and the chronological transition head:

```bash
python scripts/train_p4d_belief.py \
  --dataset-root "$P4D_DATASET" --output runs/p4d_seed0 \
  --seeds 0 1 2 3 4 --skip-classical
python scripts/train_transition.py \
  --dataset "$P4D_DATASET" \
  --belief-checkpoint runs/p4d_seed0/checkpoints/p4d/seed_0/best.pt \
  --output runs/transition_seed0
```

Collect held-out RGB-D validation observations and fit detector calibration:

```bash
python scripts/collect_calibration.py \
  --dataset "$P4D_DATASET" --tasks "$NAV_TASKS" \
  --hssd-root "$HSSD_ROOT" --navmesh-root "$NAVMESH_ROOT" \
  --limit 8 --output runs/calibration_val.jsonl
python scripts/fit_calibration.py \
  --validation-jsonl runs/calibration_val.jsonl \
  --output runs/detection_calibration.json
```

## Run

Run the event-driven N3 Agent with Grounding DINO + SAM2:

```bash
python -m evolvingnav_paper.run \
  --task n3 --world static --limit 10 \
  --dataset "$P4D_DATASET" --tasks "$NAV_TASKS" \
  --hssd-root "$HSSD_ROOT" --navmesh-root "$NAVMESH_ROOT" \
  --checkpoint runs/p4d_seed0/checkpoints/p4d/seed_0/best.pt \
  --calibration runs/detection_calibration.json \
  --output runs/n3_static_10
python -m evolvingnav_paper.verify_visual runs/n3_static_10 \
  --tasks "$NAV_TASKS" --hssd-root "$HSSD_ROOT" \
  --navmesh-root "$NAVMESH_ROOT"
```

The default controller is GPT-5.6-Luna. Use `--controller utility` for utility-only selection. Model IDs and revisions for Grounding DINO and SAM2 are in `configs/perception.yaml`.

For an N4 task directory with `public/episodes_n4.jsonl`, each private `target_motion_schedule` event supplies seconds after query (`time_s`), `target_position_xyz`, `current_state_id`, and `valid_goal_viewpoints`:

```bash
python -m evolvingnav_paper.run \
  --task n4 --world routine --limit 2 \
  --dataset "$P4D_DATASET" --tasks /absolute/path/to/n4_tasks \
  --hssd-root "$HSSD_ROOT" --navmesh-root "$NAVMESH_ROOT" \
  --checkpoint runs/p4d_seed0/checkpoints/p4d/seed_0/best.pt \
  --transition-checkpoint runs/transition_seed0/best.pt \
  --calibration runs/detection_calibration.json \
  --output runs/n4_routine_2
```

For a held-out scene, set `--task n5 --protocol n3` and point `--tasks` to its public catalog and episodes. Use `--protocol n4` with a transition checkpoint for dynamic held-out episodes.

Every run writes `policy.jsonl`, `scores.jsonl` and `summary.json` to a new output directory.

## EVOWORLD-BENCH

The repository also contains the complete evolving-world benchmark workflow.
The implementation is under `src/evoworld/`; it generates causal household
timelines, native Habitat RGB-D histories, four mobility regimes, N1-N5 task
streams, public/private audits, and Agent execution adapters.

Generate the five-scene configuration or resolve the 54-scene paper-scale
configuration without materializing files:

```bash
PYTHONPATH=.:src:scripts python scripts/generate_native_v8.py \
  --output /absolute/path/to/evoworld_v8 \
  --scale prototype

PYTHONPATH=.:src:scripts python scripts/generate_native_v8.py \
  --output /tmp/evoworld_full_plan --scale full --dry-run
```

Audit an existing dataset and export the public scene layout:

```bash
PYTHONPATH=.:src:scripts python scripts/audit_native.py \
  /absolute/path/to/evoworld_v8 --catalog-root /absolute/path/to/catalogs
PYTHONPATH=.:src:scripts python scripts/audit_paper_contract.py \
  --dataset /absolute/path/to/evoworld_v8 \
  --catalog-root /absolute/path/to/catalogs \
  --output /absolute/path/to/paper_contract_audit.json
PYTHONPATH=.:src:scripts python scripts/export_release_layout.py \
  --dataset-root /absolute/path/to/evoworld_v8 \
  --catalog-root /absolute/path/to/catalogs \
  --output /absolute/path/to/evoworld_release
```

Run the Agent on public episodes. The evaluator-private stream is consumed
only by the local execution backend:

```bash
PYTHONPATH=.:src:scripts python scripts/run_native_agent.py \
  --dataset /absolute/path/to/evoworld_v8 \
  --split test \
  --catalog-root /absolute/path/to/catalogs \
  --agent-root /absolute/path/to/EvolvingNav/code \
  --output /absolute/path/to/agent_run \
  --calibration /absolute/path/to/detection_calibration.json \
  --prior last_seen --controller utility
```

The optional train-only transition workflow is
`scripts/train_native_transition.py`; it reads train and validation streams,
never opens test records, and writes a provenance manifest with
`test_used=false`.

The benchmark code is source-only in this repository. Generated datasets,
model checkpoints, run logs, cache directories, and credentials are excluded.

## Tests

```bash
python -m pytest tests -q
```
