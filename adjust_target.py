import dataset_manage
import train
import model
import torch
import stress
import json
import numpy as np

def get_adjustment(
        n_trials=1,
    ):

    test_sets = dataset_manage.list_datasets(study="timbremetrics")
    scaling = {}

    for test_set in test_sets:

        test_embeddings, target_matrix = model.build_batch([test_set], augment_data=False, adjust_target=False)[0]

        train_sets = [
            dataset
            for dataset in dataset_manage.list_datasets(study="timbremetrics")
            if dataset_manage.get_dataset_hash(study="timbremetrics", dataset=dataset)
            != dataset_manage.get_dataset_hash(study="timbremetrics", dataset=test_set)
        ]

        scales = []

        for i in range(n_trials):

            net = train.train(datasets=train_sets)
            net.eval()

            predicted_coords = net(test_embeddings)
            predicted_matrix = torch.cdist(predicted_coords, predicted_coords, p=2)

            scales.append(stress.get_scale_mismatch(predicted_matrix, target_matrix).item())

        scaling[test_set] = [np.mean(scales), np.std(scales)]

    with open(f"data/scaling.json", "w") as f:
        json.dump(scaling, f, indent=2)

def adjust():

    with open(f"data/scaling.json", "r") as f:
        adjustment = json.load(f)

    for dataset in dataset_manage.list_datasets():

        dissimilarity_matrix = np.loadtxt(dataset_manage.get_dataset_matrixfile(study="timbremetrics", dataset=dataset))
        adjusted_dissimilarity_matrix = adjustment[dataset][0] * dissimilarity_matrix

        np.savetxt(
            dataset_manage.get_dataset_adjmatrixfile(study="timbremetrics", dataset=dataset),
            adjusted_dissimilarity_matrix,
            fmt="%.15f"
        )

if __name__ == "__main__":

    get_adjustment(n_trials=10)
    adjust()