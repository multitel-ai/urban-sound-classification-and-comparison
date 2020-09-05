#%%
import os
import shutil
import torchaudio
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets.utils import download_url, download_and_extract_archive


def one_hot(idx, num_items):
    return [(0.0 if n != idx else 1.0) for n in range(num_items)]


class ESC50(Dataset):
    base_folder = "ESC-50"
    resources = [("https://github.com/karoldvl/ESC-50/archive/master.zip", "master")]

    def __init__(self, dataset_folder, fold, transform=None, download=False):
        super().__init__()

        # Here we declare every useful path
        self.dataset_folder = dataset_folder
        self.path_to_csv = os.path.join(self.dataset_folder, "meta/esc50.csv")
        self.path_to_audio_folder = os.path.join(self.dataset_folder, "audio")
        self.path_to_melTALNet = os.path.join(self.dataset_folder, "melTALNet")

        # Downloading and extracting if needed
        if download:
            self.download()

        # Checking if the dataset exist at specified location
        if not os.path.exists(self.dataset_folder):
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        self.raw_annotations = pd.read_csv(self.path_to_csv)
        self.fold = fold

        self.transform = transform

    def download(self):

        if os.path.exists(self.dataset_folder):
            return

        # Download files
        for url, _ in self.resources:
            down_root = os.path.dirname(self.dataset_folder)
            download_and_extract_archive(
                url, download_root=down_root, filename=self.base_folder + ".zip", remove_finished=True
            )

        # Rename folder
        shutil.move(
            os.path.join(os.path.dirname(self.dataset_folder), "ESC-50-master"),
            os.path.join(os.path.dirname(self.dataset_folder), "ESC-50"),
        )

    def compute_melspec(self):
        import librosa

        if ~os.path.exists(self.path_to_melTALNet):
            os.makedirs(self.path_to_melTALNet, exist_ok=True)

        audio_list_path = [
            os.path.join(self.path_to_audio_folder, x) for x in list(pd.unique(self.raw_annotations["filename"]))
        ]

        def compute_one_mel(filename):
            wav = librosa.load(filename, sr=44100)[0]
            melspec = librosa.feature.melspectrogram(
                wav, sr=44100, n_fft=2822, hop_length=1103, n_mels=64, fmin=0, fmax=8000
            )
            logmel = librosa.core.power_to_db(melspec)
            np.save(
                os.path.join(self.path_to_melTALNet, os.path.basename(filename)[:-3] + "npy",), logmel,
            )

        _ = Parallel(n_jobs=-2)(delayed(lambda x: compute_one_mel(x))(x) for x in tqdm(audio_list_path))

    def train_validation_split(self):
        """ 
      """
        train_idx = list(self.raw_annotations[self.raw_annotations["fold"] != self.fold].index)
        val_idx = list(self.raw_annotations[self.raw_annotations["fold"] == self.fold].index)

        train_set = Subset(self, train_idx)
        val_set = Subset(self, val_idx)
        val_set.transform = None

        return train_set, val_set

    def __len__(self):
        return len(self.raw_annotations)

    def __getitem__(self, index):
        file_name = self.raw_annotations["filename"].iloc[index]
        file_path = os.path.join(self.path_to_audio_folder, file_name)

        wav, sr = torchaudio.load(file_path)
        if self.transform and self.raw_annotations["fold"].iloc[index] != self.fold:
            wav = self.transform(wav)

        label = one_hot(self.raw_annotations["target"].iloc[index], 50)

        return {
            "file_name": file_name,
            "input_vector": wav,
            "label": label,
        }


class ESC50_TALNet(ESC50):
    def __init__(self, dataset_folder, fold, transform=None, download=False):
        super().__init__(dataset_folder, fold, transform, download)

    def __getitem__(self, index):
        file_name = self.raw_annotations["filename"].iloc[index]
        file_path = os.path.join(self.path_to_melTALNet, file_name[:-3] + "npy")

        mel = np.load(file_path).transpose()
        if self.transform and self.raw_annotations["fold"].iloc[index] != self.fold:
            mel = self.transform(image=mel)["image"]

        label = np.array(one_hot(self.raw_annotations["target"].iloc[index], 50))

        return {
            "file_name": file_name,
            "input_vector": mel,
            "label": label,
        }


if __name__ == "__main__":
    import config

    dataset = ESC50(config.path_to_ESC50, 1, download=True)
    anot = dataset.raw_annotations
    train, valid = dataset.train_validation_split()
    dataset.compute_melspec()

# %%
