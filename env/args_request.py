# Args standard
# to modify (support warm start, dynamic add args item)

# args keywords
MUST_KEYS = ["learning_rate", "..."]

# args data type
DATA_TYPE = [float, "..."]

# args data domain
# Continuous: tuple(lower bound, upper bound)
# Discrete: list[val1, val2, val3]
DATA_DOMAIN = [(0, 1), [1, 2, 3]]

# args description
DATA_HELP = ["", ""]

# Dict---Key:arg keywords, Val:tuple(data type, data domain)
# Val[0]: datatype, Val[1]: data domain
ARGS_STANDARD = dict(zip(MUST_KEYS, zip(DATA_TYPE, DATA_DOMAIN, DATA_HELP)))

DEFAULT_ARGS = {
    "use_gpu": True,
    "gpu_ids": [0],
    "pre_train": False,

    "learning_rate": 0.1,
    "min_lr": 1e-9,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "nesterov": True,

    "step_size": 1,
    "gamma": 0.5 ** (1 / 10000),
    "en_alpha": 0.5,

    "loss_func": 'cross_entropy',

    "batch_limit": 99999,
    "test_batch_limit": 9999,
    "logits_batch_limit": 50,
    "loss_back": 3,
    # "mu": 0.3,

    "CE_WEIGHT": 1.0,
    "ALPHA": 1.0,
    "BETA": 8.0,
    # "T": 4.0,
    "WARMUP": 20,
    "KD_BATCH": 20,
    "KD_EPOCH": 5,

    "clusters": 10,
    "drag": 0.5,
    "threshold": 0.2,
    "step_cluster": 1,

    "mu": 0.7,
    "T": 0.5,
}
