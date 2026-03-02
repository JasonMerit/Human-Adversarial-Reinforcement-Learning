import optuna

from cleanrl_utils.tuner import Tuner

tuner = Tuner(
    script="cleanrl/ppo_cnn.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=200,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "Tron-v0": [-1, 1.5],
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_float("learning-rate", 0.0003, 0.003, log=True),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4]),
        "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8]),
        "num-steps": trial.suggest_categorical("num-steps", [5, 16, 32, 64, 128]),
        "vf-coef": trial.suggest_float("vf-coef", 0, 5),
        "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 5),
        "total-timesteps": 500000,
        "num-envs": 16,
    },
    # pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
tuner.tune(
    num_trials=20,
    num_seeds=3,
)
