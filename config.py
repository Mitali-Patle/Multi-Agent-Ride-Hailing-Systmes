"""Central configuration for the ride-hailing multi-agent simulation.

Every tunable number in the project lives here. Other modules read these
values at call time via ``import config`` / ``config.NAME`` (never
``from config import NAME``), so tests can monkeypatch any value.

Units: money is in abstract currency units, time is in ticks
(1 tick = ``TICK_MINUTES`` simulated minutes) unless a name says otherwise.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root, never hardcoded)
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODELS_SUBDIR = "models"
LOGS_SUBDIR = "logs"
CHARTS_SUBDIR = "charts"
CSV_SUBDIR = "csv"
MODEL_FILENAME = "model.zip"
TRAINING_LOG_CSV = "training_log.csv"

# --------------------------------------------------------------------------
# World and time
# --------------------------------------------------------------------------
TICK_MINUTES = 5
EPISODE_TICKS = 144  # 12 hours of operation
START_HOUR = 7.0  # clock time of tick 0 (07:00), used for display only
BASE_FARE = 150.0

# Rider arrival curve: Poisson mean per tick = base + two Gaussian bumps.
ARRIVAL_BASE_RATE = 7.0
MORNING_PEAK_TICK = 18  # 08:30
MORNING_PEAK_HEIGHT = 20.0
MORNING_PEAK_WIDTH = 10.0  # std dev in ticks
EVENING_PEAK_TICK = 126  # 17:30
EVENING_PEAK_HEIGHT = 24.0
EVENING_PEAK_WIDTH = 12.0

# --------------------------------------------------------------------------
# Platforms (the learning agents)
# --------------------------------------------------------------------------
N_PLATFORMS = 2  # 1 = monopoly, 2 = duopoly
MAX_PLATFORMS = 2
FARE_MULTIPLIERS = [0.7, 0.85, 1.0, 1.2, 1.5]
BONUS_LEVELS = [0.0, 15.0, 30.0]  # per completed ride; 30 > commission on the lowest fare, so profit can go negative
BONUS_LABELS = ["none", "low", "high"]
COMMISSION = 0.25
DEFAULT_FARE_INDEX = 2  # multiplier 1.0, used before a platform has acted
DEFAULT_BONUS_INDEX = 0
PLATFORM_NAMES = ["A", "B"]
MONOPOLY_NAME = "Monopoly"

# --------------------------------------------------------------------------
# Drivers (rule-based agents)
# --------------------------------------------------------------------------
N_DRIVERS = 50
RIDE_TICKS_MIN = 2
RIDE_TICKS_MAX = 5
MEAN_RIDE_TICKS = (RIDE_TICKS_MIN + RIDE_TICKS_MAX) / 2.0
DRIVER_EMA_ALPHA_MIN = 0.05  # per-driver belief update speed is heterogeneous
DRIVER_EMA_ALPHA_MAX = 0.25
DRIVER_INERTIA_MIN = 1.5  # utility bonus (currency/tick) for staying put
DRIVER_INERTIA_MAX = 4.0
DRIVER_LOGIT_TEMP = 2.0  # softmax temperature over platforms (currency/tick)
DRIVER_CONSIDER_PROB = 0.10  # chance an idle driver reconsiders each tick
UTILIZATION_PRIOR = 0.5  # assumed utilization of a platform with no drivers

# --------------------------------------------------------------------------
# Riders (rule-based agents)
# --------------------------------------------------------------------------
WTP_MEAN = 200.0  # willingness to pay ~ Normal(mean, std), clipped
WTP_STD = 60.0
WTP_MIN = 20.0
WTP_MAX = 600.0
WAIT_COST = 10.0  # disutility per tick of waiting
PICKUP_TICKS = 1.0  # wait when an idle driver is available immediately
MAX_WAIT_ESTIMATE = 12.0  # quoted wait (ticks) when no driver can serve soon
RIDER_LOGIT_TEMP = 15.0  # softmax temperature; <= 0 means pure best response
RIDER_PATIENCE = 3  # ticks a queued rider waits before abandoning

# --------------------------------------------------------------------------
# Shocks
# --------------------------------------------------------------------------
DEMAND_SPIKE_MULT = 2.5
DEMAND_SPIKE_TICKS = 6
OUTAGE_FRACTION = 0.4
OUTAGE_TICKS = 12
SENSOR_FAIL_TICKS = 18

# --------------------------------------------------------------------------
# Observation normalisation (observations are clipped to [0, 1])
# --------------------------------------------------------------------------
QUEUE_NORM = 40.0
ARRIVALS_NORM = 60.0
ABANDONED_NORM = 20.0
OBS_DIM = 11

# --------------------------------------------------------------------------
# Reinforcement learning
# --------------------------------------------------------------------------
REWARD_SCALE = 100.0
SEED = 42
DEMAND_STREAM_KEY = 1  # spawn key of the separate rider-arrival stream (side-by-side demo only)
N_ROUNDS = 5
STEPS_PER_ROUND = 28_800  # per platform per round (a multiple of n_steps)
TRAIN_RANDOM_SHOCK_PROB = 0.5  # chance a training episode contains a random shock
TRAIN_SHOCKS_PER_EPISODE_MAX = 2
TORCH_THREADS = 1  # small MLPs run fastest single-threaded on CPU
OPPONENT_DETERMINISTIC = True
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 1440,  # 10 full episodes per rollout, so rounds end on episode boundaries
    "batch_size": 288,
    "n_epochs": 10,
    # gamma 0.95 (~20-tick / 100-minute horizon) and lambda 0.9 cut advantage
    # variance; with 0.99 the policy collapsed to one constant price.
    "gamma": 0.95,
    "gae_lambda": 0.9,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

# Evaluation seeds are FIXED so numbers are comparable across rounds.
EVAL_SEED_START = 10_000
N_EVAL_EPISODES_TRAINING = 5  # per-round checks inside training
N_EVAL_EPISODES_FINAL = 30  # final evaluation for charts and hypotheses
SIGNIFICANCE_T = 2.0  # |Welch t| above this is treated as significant

# --------------------------------------------------------------------------
# Rule-based baseline pricing policy
# --------------------------------------------------------------------------
BASELINE_HIGH_UTIL = 0.85  # utilisation above this -> raise fare
BASELINE_LOW_UTIL = 0.45  # utilisation below this -> discount
BASELINE_QUEUE_SURGE = 0.1  # normalised queue above this -> surge fare
BASELINE_FARE_LOW = 1
BASELINE_FARE_MID = 2
BASELINE_FARE_HIGH = 3
BASELINE_FARE_SURGE = 4
BASELINE_FLEET_SHARE_FLOOR = 0.9  # bonus if own fleet < 90% of a fair share
BASELINE_BONUS_DEFAULT = 0
BASELINE_BONUS_RECRUIT = 1

# --------------------------------------------------------------------------
# Demo and telemetry
# --------------------------------------------------------------------------
SWITCH_EVENT_MIN = 5  # log an event once this many drivers have (net) switched
FARE_EVENT_MIN_JUMP = 2  # log fare changes of at least this many levels
QUEUE_EVENT_THRESHOLD = 15  # log when a queue first exceeds this
DASHBOARD_DEFAULT_SPEED = 4.0  # ticks per second in autoplay
DASHBOARD_MAX_SPEED = 10.0
DASHBOARD_MIN_SPEED = 0.5
DASHBOARD_FAST_FORWARD_TICKS = 12  # the '+1 hour' button
EVENT_LOG_MAX = 200
SCENARIO_SHOCK_TICKS = {  # tick at which each shock fires in CLI scenarios
    "demand_spike": 60,
    "driver_outage": 60,
    "sensor_failure": 60,
}
SCENARIO_ALL_SCHEDULE = [  # (tick, shock kind, target platform) for "all"
    (30, "demand_spike", None),
    (50, "driver_outage", None),
    (80, "sensor_failure", 1),
    (110, "demand_spike", None),  # overlaps with the outage below
    (112, "driver_outage", None),
    (143, "demand_spike", None),  # fired on the final tick
]

# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
COLOR_A = "#6D28D9"
COLOR_B = "#A78BFA"
COLOR_MONOPOLY = "#1E1B4B"
COLOR_BASELINE = "#9CA3AF"
COLOR_SHOCK = "#DC2626"
CHART_FONT_SIZE = 14
CHART_DPI = 150
