import os
import librosa
import numpy as np
import shutil
import torch
import torchaudio
import dataset_manage
from msclap import CLAP
from pathlib import Path
import soundfile
import stress
import augment

def _load(path, *args, **kwargs):
    y, sr = soundfile.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(y.T), sr

torchaudio.load = _load

def embed(
        study: str="timbremetrics"
    ):

    model = CLAP(version='2023', use_cuda=False)
    seen_hashes = {}

    for dataset in dataset_manage.list_datasets(study=study):

        embeddings_path = dataset_manage.get_dataset_embeddings_path(study=study, dataset=dataset)
        dataset_hash = dataset_manage.get_dataset_hash(study=study, dataset=dataset)

        if dataset_hash in seen_hashes:

            shutil.copytree(
                dataset_manage.get_dataset_embeddings_path(study=study, dataset=seen_hashes[dataset_hash]),
                embeddings_path, dirs_exist_ok=True
            )

            print(f"{dataset} identical to {seen_hashes[dataset_hash]}: copied embeddings")
            continue

        seen_hashes[dataset_hash] = dataset

        for audio_file in dataset_manage.get_dataset_audiofiles(study=study, dataset=dataset):

            if os.path.isdir("./cache/embed"): shutil.rmtree("./cache/embed")
            os.makedirs("./cache/embed", exist_ok=True)
            audio, sr = librosa.load(audio_file, sr=model.args.sampling_rate)

            batch_paths, batch_names = [], []
            for variant in augment.get_augment_meshgrid():

                modified_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=variant["pitch"])
                modified_audio = np.pad(modified_audio, (int(variant["delay"] * sr), 0))
                modified_audio *= variant["volume"]

                name = f"{Path(audio_file).stem}_d={variant['delay']}_v={variant['volume']}_p={variant['pitch']}"
                path = f"./cache/embed/{name}.wav"
                soundfile.write(path, modified_audio, sr, subtype="FLOAT")

                batch_paths.append(path)
                batch_names.append(name + ".npy")

            embeddings = model.get_audio_embeddings(batch_paths).cpu().detach().numpy()
 
            for embedding, name in zip(embeddings, batch_names):
                np.save(os.path.join(embeddings_path, name), embedding)

        print(f"{dataset} done embedding")

# def pca_compress(x, n_components=2):

#     x_centered = x - x.mean(dim=0, keepdim=True)
#     U, S, V = torch.pca_lowrank(x_centered, q=n_components)
#     return x_centered @ V[:, :n_components]

def embedding_analyze(
        study: str="timbremetrics"
    ):

    normal_loss: torch.Tensor = 0
    triplet_loss: torch.Tensor = 0
    NDCG_loss: torch.Tensor = 0

    n_normal = 0
    n_triplet = 0
    n_NDCG = 0

    for dataset in dataset_manage.list_datasets(study=study):

        embedding_files = dataset_manage.get_dataset_embeddingfiles(study=study, dataset=dataset)

        dataset_embeddings = torch.stack([
            torch.from_numpy(np.load(file)).float()
            for file in embedding_files
            if augment.is_no_augment(file)
        ])

        # dataset_embeddings = pca_compress(dataset_embeddings, n_components=2)

        predicted_matrix = torch.cdist(dataset_embeddings, dataset_embeddings, p=2)

        target_matrix = torch.from_numpy(
            np.loadtxt(
                dataset_manage.get_dataset_matrixfile(study=study, dataset=dataset)
            )
        ).float()

        l, n = stress.stress_loss(predicted_matrix, target_matrix, rescale_type="normalize", loss_type="MAE")
        normal_loss, n_normal = normal_loss + l * n, n_normal + n
        l, n = stress.stress_loss(predicted_matrix, target_matrix, rescale_type="normalize", loss_type="triplet")
        triplet_loss, n_triplet = triplet_loss + l * n, n_triplet + n
        l, n = stress.stress_loss(predicted_matrix, target_matrix, rescale_type="normalize", loss_type="NDCG")
        NDCG_loss, n_NDCG = NDCG_loss + l * n, n_NDCG + n

    print(f"MAE (normal): {normal_loss.item() / n_normal}")
    print(f"triplet: {triplet_loss.item() / n_triplet}")
    print(f"NDCG: {NDCG_loss.item() / n_NDCG}")

if __name__ == "__main__":

    #embed(study="timbremetrics")

    embedding_analyze(study="timbremetrics")