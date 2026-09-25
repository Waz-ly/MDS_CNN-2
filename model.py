import torch.nn as nn
import torch
import numpy as np
import augment
import dataset_manage
import stress

class EmbeddingNet(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()

        self.layer_dims = layer_dims

        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        layers.pop()
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def build_batch(
        datasets: list[str],
        augment_data: bool=False,
        adjust_target: bool=False,
        augment_mesh: list[dict] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

    augment_mesh = augment.get_augment_meshgrid() if augment_mesh is None else augment_mesh

    study = "timbremetrics"
    n_variants = len(augment_mesh)
    batches = []

    for dataset in datasets:

        dataset_embeddings = torch.stack([
            torch.from_numpy(np.load(file)).float()
            for file in dataset_manage.get_dataset_embeddingfiles(study=study, dataset=dataset)
            if (augment_data and augment.in_variant_mesh(file, augment_mesh)) or augment.is_no_augment(file)
        ])

        target_matrix = torch.from_numpy(np.loadtxt(
            dataset_manage.get_dataset_matrixfile(study=study, dataset=dataset)
            if not adjust_target
            else dataset_manage.get_dataset_adjmatrixfile(study=study, dataset=dataset)
        )).float()

        if augment_data: target_matrix = torch.repeat_interleave(torch.repeat_interleave(target_matrix, n_variants, dim=0), n_variants, dim=1)

        batches.append([dataset_embeddings, target_matrix])

    return batches

def model_stress(
        model: nn.Module,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
        rescale_type: str = "none",
        loss_type: str = "MAE",
    ):

    total_loss: torch.Tensor = 0.0
    total_items = 0

    for [embeddings, target_matrix] in batch:

        predicted_coords = model(embeddings)
        predicted_matrix = torch.cdist(predicted_coords, predicted_coords, p=2)
        loss, n_items = stress.stress_loss(predicted_matrix, target_matrix, rescale_type, loss_type)

        total_loss += loss * n_items
        total_items += n_items

    mean_loss = total_loss / total_items
    return mean_loss, total_items

