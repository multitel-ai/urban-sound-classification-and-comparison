#%%
import os
import torchaudio
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets.utils import download_url, download_and_extract_archive


def one_hot(idx, num_items):
    return [(0.0 if n != idx else 1.0) for n in range(num_items)]


class UrbanSound8K(Dataset):
    base_folder = "UrbanSound8K"
    resources = [("https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz", "9aa69802bbf37fb986f71ec1483a196e")]

    def __init__(self, dataset_folder, fold, transform=None, download=False):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.path_to_csv = os.path.join(self.dataset_folder, "metadata/UrbanSound8K.csv")
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

    def download(self):

        if os.path.exists(self.dataset_folder):
            return

        # Download files
        for url, md5 in self.resources:
            down_root = os.path.dirname(self.dataset_folder)
            download_and_extract_archive(
                url, download_root=down_root, filename=self.base_folder + ".tar.gz", md5=md5, remove_finished=True
            )

    def compute_melspec(self):
        import librosa

        if ~os.path.exists(self.path_to_melTALNet):
            os.makedirs(self.path_to_melTALNet, exist_ok=True)
            for i in range(1, 11):
                os.makedirs(os.path.join(self.path_to_melTALNet, "fold" + str(i)), exist_ok=True)

        audio_list_path = []
        for index in range(len(self)):
            file_name = self.raw_annotations["slice_file_name"].iloc[index]
            file_path = os.path.join(
                os.path.join(self.path_to_audio_folder, "fold" + str(self.raw_annotations["fold"].iloc[index])),
                file_name,
            )
            audio_list_path.append(file_path)

        def compute_one_mel(filename):
            wav = librosa.load(filename, sr=44100)[0]
            # wav = librosa.util.pad_center(wav, 178017)
            if 178017 - wav.shape[0] != 0:
                wav = np.concatenate((np.zeros((178017 - wav.shape[0])), wav))
            melspec = librosa.feature.melspectrogram(
                wav, sr=44100, n_fft=2822, hop_length=1103, n_mels=64, fmin=0, fmax=8000
            )
            logmel = librosa.core.power_to_db(melspec)
            np.save(
                os.path.join(
                    self.path_to_melTALNet,
                    os.path.split(os.path.split(filename)[0])[1],
                    os.path.basename(filename)[:-3] + "npy",
                ),
                logmel,
            )

        _ = Parallel(n_jobs=-2)(delayed(lambda x: compute_one_mel(x))(x) for x in tqdm(audio_list_path))

    def get_max_lenght(self):
        import librosa

        audio_list_path = []
        for index in range(len(self)):
            file_name = self.raw_annotations["slice_file_name"].iloc[index]
            file_path = os.path.join(
                os.path.join(self.path_to_audio_folder, "fold" + str(self.raw_annotations["fold"].iloc[index])),
                file_name,
            )
            audio_list_path.append(file_path)

        maxi = 0
        mini = 200000
        for i in tqdm(range(len(audio_list_path))):
            wav = librosa.load(audio_list_path[i], sr=44100)[0]
            lenghtofwav = wav.shape[0]
            if lenghtofwav > maxi:
                maxi = lenghtofwav
            if lenghtofwav < mini:
                mini = lenghtofwav

        print("maxi = ", maxi)
        print("mini = ", mini)
        # 44100 Hz : max 178017 min 2205

    def __getitem__(self, index):
        file_name = self.raw_annotations["slice_file_name"].iloc[index]
        file_path = os.path.join(
            os.path.join(self.path_to_audio_folder, "fold" + str(self.raw_annotations["fold"].iloc[index])), file_name
        )

        wav, sr = torchaudio.load(file_path)
        if self.transform and self.raw_annotations["fold"].iloc[index] != self.fold:
            wav = self.transform(wav)

        label = np.array(one_hot(self.raw_annotations["classID"].iloc[index], 10))

        return {
            "file_name": file_name,
            "input_vector": wav,
            "label": label,
        }


class UrbanSound8K_TALNet(UrbanSound8K):
    def __init__(self, dataset_folder, fold, transform=None, download=False):
        super().__init__(dataset_folder, fold, transform, download)

    def __getitem__(self, index):
        file_name = self.raw_annotations["slice_file_name"].iloc[index]
        file_path = os.path.join(
            self.path_to_melTALNet, "fold" + str(self.raw_annotations["fold"].iloc[index]), file_name[:-3] + "npy"
        )

        mel = np.load(file_path).transpose()
        if self.transform and self.raw_annotations["fold"].iloc[index] != self.fold:
            mel = self.transform(image=mel)["image"]

        label = np.array(one_hot(self.raw_annotations["classID"].iloc[index], 10))

        return {
            "file_name": file_name,
            "input_vector": mel,
            "label": label,
        }


if __name__ == "__main__":
    # import config
    path = "/home/imagedpt/Augustin/DCASE/data/UrbanSound8K"
    dataset = UrbanSound8K(path, 1)
    anot = dataset.compute_melspec()
    # dataset.get_max_lenght()

# %%
