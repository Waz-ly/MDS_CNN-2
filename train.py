import torch
import numpy as np
import random
import model
import dataset_manage
import os
import csv

def train(
        datasets: list[str]=[None],
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        in_dim: int = 1024,
        hidden_dim: int = 32,
        n_hidden: int = 3,
        out_dim: int = 8,
        n_epochs: int = 1000,
        augment_data: bool = False,
        adjust_target: bool = False,
        augment_mesh: list[dict] = None,
    ) -> model.EmbeddingNet:

    training_batch = model.build_batch(datasets, augment_data=augment_data, adjust_target=adjust_target, augment_mesh=augment_mesh)

    layer_dims = [in_dim] + [hidden_dim] * n_hidden + [out_dim]
    net = model.EmbeddingNet(layer_dims=layer_dims)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):

        optimizer.zero_grad()
        loss, _ = model.model_stress(net, training_batch, rescale_type="none", loss_type="MAE")
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.6f}")

    print("done training...")

    return net

def test(
        net: model.EmbeddingNet,
        datasets: list[str]=[None],
    ):

    net.eval()
    test_batch = model.build_batch(datasets=datasets, augment_data=False, adjust_target=False)

    def get_result(**kwargs):
        result = model.model_stress(net, test_batch, rescale_type='normalize', **kwargs)
        return (result[0].item(), result[1])

    return {
        "MAE": get_result(loss_type="MAE"),
        "triplet": get_result(loss_type="triplet"),
        "NDCG": get_result(loss_type="NDCG"),
    }

def trial(
        n_datasets: int | None = None,
        trials = 1,
        seed = 1,
        **model_params
    ):
    """
    n_datasets: int | None - number of datasets to create score with, if None all are used
    trials: int - number of times each dataset is scored
    seed: int - random number seed
    """

    rng = random.Random(seed)

    test_sets = dataset_manage.list_datasets(study="timbremetrics")
    scored_sets = test_sets if n_datasets is None else rng.sample(test_sets, n_datasets)

    results: list[dict[str, tuple[float, int]]] = []

    for test_set in scored_sets:

        train_sets = [
            dataset
            for dataset in dataset_manage.list_datasets(study="timbremetrics")
            if dataset_manage.get_dataset_hash(study="timbremetrics", dataset=dataset)
            != dataset_manage.get_dataset_hash(study="timbremetrics", dataset=test_set)
        ]

        for i in range(trials):

            net = train(datasets=train_sets, **model_params)
            results.append(test(net, datasets=[test_set]))

    return results

def save_results(results, filename="trial_results"):

    METRICS = ["MAE", "triplet", "NDCG"]

    def format_average(values: np.ndarray, weights: np.ndarray) -> tuple[str, str, str]:
        mean, weights_sum = np.average(values, weights=weights, returned=True)
        variance = np.average((values - mean) ** 2, weights=weights)
        return f"{mean:.4f}", f"{variance:.4f}", f"{int(weights_sum):d}"

    # Build one [value, weight] pair per metric, per result
    data = np.array([
        [x for metric in METRICS for x in result[metric]]
        for result in results
    ])

    # Compute (mean, variance, n) for each metric
    averages = [
        format_average(data[:, 2 * i], data[:, 2 * i + 1])
        for i in range(len(METRICS))
    ]

    labels = [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "var", "n")]
    row = [value for triple in averages for value in triple]

    with open(os.path.join("data", f"{filename}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(labels)
        writer.writerow(row)

if __name__ == "__main__":

    import json
    import augment

    aug_mesh = augment.get_augment_meshgrid(augment.BEST_VARIANTS)

    for i in [2, 4, 8]:

        with open(f"data/tuning_{i}D.json", "r") as f:
            model_params = json.load(f)["params"]

        results = trial(n_datasets=None, augment_data=True, augment_mesh=aug_mesh, adjust_target=True, trials=1, **model_params)
        save_results(results, f"opt_{i}D")

    raise Exception("stop")

    TEST_SET = ["Siedenburg2016_e2set2"]
    test_set_hashes = [
        dataset_manage.get_dataset_hash(study="timbremetrics", dataset=dataset)
        for dataset in TEST_SET
    ]

    TRAIN_SET = [
        dataset
        for dataset in dataset_manage.list_datasets(study="timbremetrics")
        if dataset_manage.get_dataset_hash(study="timbremetrics", dataset=dataset) not in test_set_hashes
    ]

    net = train(TRAIN_SET)
    print(test(net, TEST_SET))
