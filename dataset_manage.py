import os
from pathlib import Path
import hashlib

MATRIX_EXTENSION = "_dissimilarity_matrix.txt"
ADJMATRIX_EXTENSION = "_adj_dissimilarity_matrix.txt"
SOUNDS_EXTENSION = "_sounds"
EMBEDDINGS_EXTENSION = "_embeddings"
AUDIO_EXTENSIONS = [".aiff", ".wav"]
EMBED_EXTENSIONS = [".npy"]

def list_datasets(
        study: str="timbremetrics",
    ):

    dir_path = Path(study) 
    return sorted([f.name for f in dir_path.iterdir() if f.is_dir()])

def get_dataset_sounds_path(
        study: str="timbremetrics",
        dataset: str="None",
    ):

    dataset_path = os.path.join(study, dataset)
    sounds_path = os.path.join(dataset_path, dataset + SOUNDS_EXTENSION)

    if not os.path.isdir(sounds_path):
        raise Exception(f"dataset sound directory {sounds_path} not found")

    return sounds_path

def get_dataset_embeddings_path(
        study: str="timbremetrics",
        dataset: str="None",
    ):

    dataset_path = os.path.join(study, dataset)
    embeddings_path = os.path.join(dataset_path, dataset + EMBEDDINGS_EXTENSION)

    os.makedirs(embeddings_path, exist_ok=True)

    return embeddings_path

def get_dataset_matrixfile(
        study: str="timbremetrics",
        dataset: str="None",
    ):

    dataset_path = os.path.join(study, dataset)
    matrixfile = os.path.join(dataset_path, dataset + MATRIX_EXTENSION)

    if not os.path.isfile(matrixfile):
        raise Exception(f"matrix {matrixfile} not found")

    return matrixfile

def get_dataset_adjmatrixfile(
        study: str="timbremetrics",
        dataset: str="None",
    ):

    dataset_path = os.path.join(study, dataset)
    adjmatrixfile = os.path.join(dataset_path, dataset + ADJMATRIX_EXTENSION)

    return adjmatrixfile

def get_dataset_audiofiles(
        study: str="timbremetrics",
        dataset: str="None",
    ):

    sounds_path = get_dataset_sounds_path(study=study, dataset=dataset)

    return sorted([
        os.path.join(sounds_path, audiofile_name)
        for audiofile_name in os.listdir(sounds_path)
        if any([audiofile_name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS])
    ])

def get_dataset_embeddingfiles(
        study: str="timbremetrics",
        dataset: str="None",
    ):

    embeddings_path = get_dataset_embeddings_path(study=study, dataset=dataset)

    return sorted([
        os.path.join(embeddings_path, embeddingfile_name)
        for embeddingfile_name in os.listdir(embeddings_path)
        if any([embeddingfile_name.lower().endswith(ext) for ext in EMBED_EXTENSIONS])
    ])

def get_dataset_hash(
        study: str="timbremetrics",
        dataset: str="None",
    ):

    h = hashlib.md5()
    
    for audio_file in get_dataset_audiofiles(study=study, dataset=dataset):

        with open(audio_file, "rb") as f:
            h.update(f.read())

    return h.hexdigest()