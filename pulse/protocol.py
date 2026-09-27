"""Shared experimental constants (paper Sec. 'Experiments' and App. protocols)."""

P_OBSERVED = 8
HIDDEN_COUNTS = (0, 4, 8, 12)
NETWORKS_PER_SETTING = 24

MAIN_RADIUS = 0.76          # integrated spectral radius of main networks
HIGH_RADIUS = 0.90          # near-critical stress networks
ETA = 0.8                   # event-generation suppression fraction
N_TARGET_CANDIDATES = 8

TRIAL_WINDOW = 6.0          # seconds per pulse/sham window
BURN_IN = 12.0              # shared immigrant burn-in for both trial arms
PASSIVE_DURATION = 2400.0   # seconds of passive recording for baseline rates

INIT_BLOCKS = 40            # initial pulse/sham blocks per target
BATCH_BLOCKS = 40           # acquisition batch size B
POOL_PER_TARGET = 2560      # stored evaluation pool per target
BUDGETS = (320, 640, 1280, 2560, 5120)

BOOTSTRAP_RESAMPLES = 5000

# Network seed ranges (App. 'Network generation')
def main_seed(q, u):
    return 13000 + 100 * q + u

OPERATOR_SEEDS = range(22000, 22012)     # operator recovery experiments
RECRUITMENT_SEEDS = range(24000, 24016)  # direct hidden recruitment
WINDOW_SEEDS = range(25000, 25012)       # near-critical response windows
VALIDATION_SEEDS = range(26000, 26004)   # direct intervention validation
TIMING_SEED_BASE = 31000                 # + participant index u in 1..74
REGIME_SEEDS = range(33000, 33008)       # feedback-strength window grid
EQUIVALENCE_SEED_BASE = 51000            # + family index f in 0..23
EQUIVALENCE_TRIAL_BASE = 61000           # + 10 f + variant index v
