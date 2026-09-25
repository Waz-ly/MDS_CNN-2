import optuna
import numpy as np
import train
import json

def objective(trial: optuna.Trial, n_datasets: int, out_dim) -> float:

    def hidden_dim_bounds(out_dim):

        k = int(np.log2(out_dim))
        lo = max(4, k - 1)   # never below 16, at least ~out_dim/2
        hi = max(9, k + 2)   # never below 512, up to ~4x out_dim

        return lo, hi

    model_params = {
        "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "hidden_dim": 2 ** trial.suggest_int("log2_hidden_dim", *hidden_dim_bounds(out_dim)),
        "n_hidden": trial.suggest_int("n_hidden", 1, 4),
        "n_epochs": trial.suggest_int("n_epochs", 1000, 3000, step=200),
        "out_dim": out_dim
    }

    scores = train.trial(n_datasets=n_datasets, augment_data=False, adjust_target=True, **model_params)
    mae_scores = [score["MAE"][0] for score in scores]
    mae_weights = [score["MAE"][1] for score in scores]
    final_score = np.average(mae_scores, weights=mae_weights)

    return final_score

def optimize(
    n_trials: int = 50,
    n_folds: int = None,
    out_dim: int = None,
    seed: int = 42,
) -> dict:

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )

    study.optimize(
        lambda t: objective(t, n_folds, out_dim),
        n_trials=n_trials,
    )

    best_params = study.best_params
    print(f"Best params from search: {best_params}")
    print(f"Search score (normalize/MAE, {n_folds} folds): "
          f"{study.best_value:.4f}")

    return {
        "params": best_params,
        "search_score": study.best_value,
        "study": study,
    }

def save_result(result: dict, path: str = "data/tuning_result.json") -> None:

    params = result["params"]
    params["hidden_dim"] = 2 ** params["log2_hidden_dim"]
    params.pop("log2_hidden_dim")

    to_save = {
        "params": result["params"],
        "search_score": result["search_score"],
    }
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Saved to {path}")

if __name__ == "__main__":

    # 10 of 21 folds for validation

    for i in [32, 64, 128, 256]:

        result = optimize(n_trials=50, n_folds=10, out_dim=i)
        save_result(result, f"data/tuning_{i}D.json")
