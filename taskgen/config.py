"""Every setting behind a keep/discard verdict (not to be edited by the agent)."""

SEED = 0

# Splits: train has one timestamp per timedelta from the val cutoff back to the first row
MAX_TRAIN_TIMESTAMPS = 10_000
MIN_ROWS = 20  # label rows per split
MIN_MINORITY = 5  # labels per split other than the most common one (binary: the rarer class)

# Leakage: checked at the latest train timestamps and the val and test cutoffs
LEAKAGE_TRAIN_TIMESTAMPS = 3
STRICT_PAST_TIMESTAMPS = 3  # latest task timestamps with rows exactly at them, where < timestamp is checked

# Novelty
DUPLICATE_CORR = 0.95  # |Spearman| of two tasks' labels on their shared (timestamp, entity) rows
DUPLICATE_MIN_OVERLAP = 0.5  # share of rows two tasks must have in common to be compared

# Learnability: kept if the confidence interval of the model's lift over the best baseline lies above 0
LEARNABILITY_SAMPLE_SIZE = 10_000  # max label rows: sampled from train to fit, from val + test to score
DFS_MAX_DEPTH = 2  # featurizer: foreign-key hops aggregated into features
PREDICTOR = "lightgbm"  # fitted on the featurizer's features: a key of PREDICTOR_PARAMS
PREDICTOR_PARAMS = {  # else defaults; importance_type: top_features are shares of total split gain
    "lightgbm": dict(
        deterministic=True, force_row_wise=True, random_state=SEED, verbose=-1, importance_type="gain"
    ),
    "xgboost": dict(random_state=SEED, verbosity=0, importance_type="total_gain"),
    # foundation models, on CPU for reproducible verdicts
    "tabicl": dict(random_state=SEED, device="cpu"),  # TabICLv2
    "kumo-tabular": dict(size="small", random_state=SEED, device="cpu"),  # NVIDIA's; medium, large on GPUs
}
BOOTSTRAP_RESAMPLES = 200  # resamples of the scored rows that give the lift's confidence interval
LIFT_CONFIDENCE = 0.95  # level of that interval
LEAKAGE_SCORE = {"auroc": 0.98, "r2": 0.95}  # a model this accurate suggests a column leaks the label
