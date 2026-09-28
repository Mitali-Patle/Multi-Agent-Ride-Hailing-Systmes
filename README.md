# Competitive Multi-Agent Ride-Hailing Simulation

Two self-interested ride-hailing platforms learn, with independent PPO, to set
fares and driver bonuses in a shared market. They compete for a finite pool of 50
rule-based drivers and a stream of Poisson-arriving riders. The same market can
be run with a single platform (a monopoly) that has the same drivers and demand.

**Research question.** How does competition between two learning platforms
that share drivers and riders affect rider welfare, driver earnings, platform
profit and market efficiency, compared with an equally resourced monopoly?

| Hypothesis | Statement |
|---|---|
| H1 | Duopoly gives lower fares and shorter waits than monopoly. |
| H2 | Competition does not straightforwardly raise driver earnings (bonus wars erode platform profit without proportionally helping drivers). |
| H3 | Total welfare (rider surplus + driver earnings + platform profit) is lower under duopoly than under an equally resourced monopoly. |

Everything is free and open source, runs on a CPU, and is seeded end to end.

---

## Architecture

```
config.py                  every tunable number (no magic numbers elsewhere)
sim/                       the shared world (no RL in here)
  drivers.py               50 individual driver agents: beliefs (EMA), logit switching with inertia, rides, outages
  riders.py                rider arrivals (daily Poisson curve), quotes, logit choice vs outside option, FIFO queues
  shocks.py                ShockManager: demand spike, driver outage, rival-price sensor failure (stackable)
  market.py                MarketSim: runs one tick in a fixed 6-step order; supports 1 or 2 platforms
  metrics.py               per-tick records and per-episode welfare summaries
envs/
  platform_env.py          Gymnasium env = one platform's partial view; rival supplied as a callable policy
training/
  baselines.py             fixed rule-based surge policy (bootstrap opponent and evaluation yardstick)
  common.py                PPO factory, paths, logging helpers
  policies.py              FrozenPolicy: NumPy snapshot of a PPO actor (self-play opponent, fast inference)
  rollout.py               play whole episodes on fixed seeds
  train_duopoly.py         alternating self-play, 2 independent PPO models
  train_monopoly.py        single-platform PPO (the ablation)
  evaluate.py              final evaluation -> CSVs + charts
  hypotheses.py            Welch t-tests and data-driven verdicts for H1-H3
  charts.py                projector-friendly matplotlib charts
demo/
  live_sim.py              inference-only live simulation + event log (shared by both demos, unit-tested)
  dashboard.py, panels.py  Streamlit dashboard
  run_scenario.py          headless CLI demo with terminal telemetry
tests/                     pytest suite (49 tests)
results/                   models/, logs/, csv/, charts/ (created at runtime)
```

**Data flow.** `sim.MarketSim` holds the world. `envs.PlatformEnv` wraps it so
that one platform is the learner and the other platform is a fixed callable.
`training/train_*.py` train Stable-Baselines3 PPO on those envs and save models
to `results/models/`. `training/evaluate.py` loads the saved models, plays fixed
seeds and writes charts. `demo/` loads the same saved models and only runs
inference, never training.

### One tick (5 simulated minutes), in order

1. Apply shocks: finished shocks expire (outage drivers return), then new shocks start.
2. Platforms post their actions: fare level (5 options) and driver bonus (3 options).
3. Drivers update their earnings beliefs from public information. Some idle drivers may switch platform (duopoly only).
4. Queued riders who have run out of patience abandon. The remaining queue is served first, then new riders choose a platform or no ride and are matched or queued.
5. Rides advance. Finished rides pay out: the driver gets (1 − commission)·fare + bonus, and the platform gets commission·fare − bonus.
6. Metrics are recorded and observations refreshed.

### RL setup

* **Observation (11 values in [0, 1]):** own idle, busy and queued counts; own fare and bonus; the rival's last *public* fare and bonus; a `rival_feed_ok` flag; arrivals last tick; own abandonments last tick; time of day. The rival's drivers and internals are never visible (partial observability).
* **Action:** `MultiDiscrete([5, 3])` for the fare multiplier and the bonus level.
* **Reward:** the platform's profit this tick / `REWARD_SCALE`.
* **Learner:** one independent PPO `MlpPolicy` (2×64) per platform, random initialisation, no shared weights.
* **Self-play:** in round 1, A trains against the rule baseline, then B trains against A. In later rounds, each trains against the other's latest frozen snapshot. After every round, both are evaluated on the **same fixed seeds** against the baseline and against each other.
* Training episodes randomly include shocks, so the policies have seen demand spikes, outages and blind rival feeds before the live demo.

---

## Setup

Requires Python 3.10+ (tested on 3.14, macOS). Works on Windows, macOS and Linux.

```bash
python3 -m venv .venv
```

(On Windows use `py -m venv .venv`.) Activate it with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\activate` on Windows. Then:

```bash
pip install -r requirements.txt
```

All commands below are run from the project root with the virtual environment active.

## Commands, in order

1. Run the tests (about 15 s):

```bash
python -m pytest
```

2. Train the duopoly (alternating self-play; about 2 min on an Apple M5, a few minutes on a typical laptop):

```bash
python training/train_duopoly.py
```

Options: `--rounds N --steps N --seed N` (defaults come from `config.py`).

3. Train the monopoly, with the same total timesteps as one duopoly platform (about 45 s):

```bash
python training/train_monopoly.py
```

4. Evaluate and make the charts (about 15 s):

```bash
python training/evaluate.py
```

This writes `results/charts/learning_curves.png`, `learning_proof.png`, `monopoly_vs_duopoly.png` and `hypotheses_summary.csv`.

5. Launch the live dashboard:

```bash
streamlit run demo/dashboard.py
```

6. Run the headless CLI scenarios (backup demo):

```bash
python demo/run_scenario.py --scenario all --mode duopoly --seed 7
```

```bash
python demo/run_scenario.py --scenario sensor_failure --mode duopoly --seed 7
```

```bash
python demo/run_scenario.py --scenario all --mode monopoly --seed 7
```

Scenarios: `steady`, `demand_spike`, `driver_outage`, `sensor_failure`, `all`. Add `--policy baseline` to run the rule-based policies, or `--quiet` to hide the per-tick table.

Optional: monitor training in TensorBoard:

```bash
tensorboard --logdir results/logs/tensorboard
```

## Dashboard controls

| Control | What it does |
|---|---|
| Market | Switch between the duopoly (2 trained platforms) and the monopoly (1 trained platform). |
| Seed + Reset | Restart the 12-hour day with a new random seed. |
| Step 1 tick | Advance 5 simulated minutes. |
| Autoplay + speed | Run continuously (0.5 to 10 ticks per second). The shock buttons keep working while it runs. |
| Use rule-based baseline | Fallback if no trained models exist. The page labels the policy source. |
| **Demand spike** | *A concert just ended:* arrivals ×2.5 for 30 minutes. |
| **Driver outage** | *Heavy rain:* 40 % of drivers log off for 1 hour. Busy drivers finish their current ride first. |
| **Sensor failure** | *Rival price feed down:* the chosen platform's view of the rival's prices freezes at the last known values, and its `rival_feed_ok` flag drops to 0 for 90 minutes. Disabled in monopoly mode, which has no rival. |

Panels: driver states per platform, fares and bonuses over time, queues and abandonments, cumulative profit, market share, rider welfare and driver pay, and a timestamped **event log** (shocks starting and ending, bonus changes, big fare moves, driver migrations, queue alerts, loss-making ticks). Active shocks show as a red banner and as shaded bands on every chart.

---

## Results

These numbers come from the default config and seed 42, with 30 fixed evaluation seeds. They were measured on an Apple M5 CPU and are regenerated by the commands above. Because everything is seeded, a rerun gives exactly the same numbers. See `results/charts/` for the charts and `hypotheses_summary.csv` for all the statistics.

**Training time:** duopoly 103 s, monopoly 43 s, 2.5 min in total (5 rounds × 28,800 steps per platform).

**Learning proof.** Mean episode profit against the same rule-baseline opponent, on the same seeds:

| | Untrained (random init) | Trained PPO | Rule baseline |
|---|---:|---:|---:|
| Platform A | 4,178 | **32,924** | 26,282 |
| Platform B | 5,907 | **33,575** | 26,179 |
| Monopoly | 26,344 | **54,058** | 52,451 |

**What the agents learned.** Duopoly platforms fight a price war off-peak (0.7× fare) and raise prices to 1.2× when queues build at the peaks. The monopolist charges 1.0× off-peak and 1.2× at peaks. **No platform ever paid a driver bonus**, because under this calibration a bonus costs more than the drivers it attracts are worth.

**Monopoly vs duopoly (trained policies, means over 30 seeds):**

| Metric | Monopoly | Duopoly | Hypothesis verdict |
|---|---:|---:|---|
| Average fare | 170.9 | 135.8 (−21 %) | H1: **partly supported** |
| Average wait (min) | 6.5 | 9.2 (+42 %) | fares are lower, but waits are *longer*: the fleet is split in two, so one platform can have a queue while the other has idle drivers |
| Abandonment rate | 0.0 % | 0.5 % | |
| Earnings per driver | 3,243 | 3,111 (−4 %) | H2: **supported**. Driver pay *fell* along with fares. No bonus war happened: the effect comes through fares, not bonuses. |
| Platform profit (total) | 54,058 | 51,847 (−4 %) | |
| Total welfare | 281,784 | 310,188 (+10 %) | H3: **not supported**. Lower prices served more riders, and rider surplus rose 57 % (see the logit caveat below). |

**Shock robustness** (trained duopoly, change vs no shock, 30 seeds): a peak-hour driver outage costs each platform about 1.1k–1.3k profit and adds 1.6 percentage points of abandonment. A demand spike *raises* profit. A sensor failure changes profit by less than 1 %, because the learned policies rely mostly on their own queue, their own utilisation and the time of day rather than on the rival's price.

---

## Design decisions and scope

* **One zone.** No spatial model: waits come from each platform's idle drivers and queue length.
* **Two of the five planned actuators:** fare level and driver bonus. Surge caps, rider discounts and recruitment spend are future extensions.
* **Bonus levels are [0, 15, 30] per ride.** The top bonus is larger than the commission on the cheapest fare (0.25 × 105 = 26.25), so a platform *can* lose money on every ride. This is intentional: it lets a bonus war actually hurt.
* **Independent PPO is the primary method.** MADDPG (centralised critic) is the documented fallback if independent PPO failed to converge. It was not needed and is **not implemented**.
* **PPO settings:** `gamma = 0.95`, `gae_lambda = 0.9` and `n_epochs = 10`. With the default `gamma = 0.99`, the policies collapsed to a single constant price. The shorter horizon (about 100 simulated minutes) reduced advantage noise enough for state-dependent pricing to emerge.
* **Demand calibration:** peaks (about 30 arrivals per tick at 17:30) exceed fleet capacity (about 14 rides per tick), and off-peak demand is well below it. This makes surge pricing and driver supply matter, as they do in real markets. With a flatter curve, every policy converged to one fixed price.
* **Accounting:** revenue is booked when a ride completes. At closing time, rides still in progress are completed and paid, so every matched ride is settled. Riders who abandon incur the waiting cost they already paid.

## Known limitations

* **Logit "variety" effect.** Riders use a plain logit over platforms plus "no ride". Adding a second platform therefore slightly raises the chance that a rider rides at all, even at identical prices (the IIA property; it can be read as idiosyncratic brand taste). This favours the duopoly in rider surplus and welfare (H3). A nested logit would remove it.
* **Driver supply is fixed.** Drivers never leave the market for low pay (only outages take them offline). Drivers bear no costs, so driver earnings are gross transfers.
* **Small, discrete action space and short training.** The policies are good responses, not proven equilibria. Results come from one training seed. Retraining with other seeds (`--seed`) would show the variance across runs.
* **Evaluation uses deterministic (arg-max) actions**, and the self-play opponent is also deterministic.
* **p-values** use a normal approximation to Welch's t (fine for 30 episodes, avoids a SciPy dependency).
* Streamlit re-renders the whole page each tick, so very high autoplay speeds are limited by the browser.

## Files to read first

1. `config.py`: every parameter in one place.
2. `sim/market.py`: the tick order and how everything connects.
3. `envs/platform_env.py`: how a two-player market becomes a single-agent RL problem.
4. `training/train_duopoly.py`: alternating self-play.
5. `demo/live_sim.py`: what the live demo shows and how the event log is produced.
