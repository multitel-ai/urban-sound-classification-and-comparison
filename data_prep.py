import argparse
from prepare_data.sonycust import SONYCUST
from prepare_data.esc50 import ESC50
from prepare_data.urbansound8k import UrbanSound8K
import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", help="download dataset", action="store_true")
    parser.add_argument("--mel", help="precompute mels for TALNet", action="store_true")
    args = parser.parse_args()

    if args.download:
        print("Downloading Dataset")
        dataset = SONYCUST(config.path_to_SONYCUST, "coarse", download=True)
        dataset = ESC50(config.path_to_ESC50, fold=1, download=True)
        dataset = UrbanSound8K(config.path_to_UrbanSound8K, fold=1, download=True)

    else:
        dataset = SONYCUST(config.path_to_SONYCUST, "coarse")
        dataset = ESC50(config.path_to_ESC50, fold=1)
        dataset = UrbanSound8K(config.path_to_UrbanSound8K, fold=1)

    if args.mel:
        print("Computing mel spectrograms for TALNet")
        dataset.compute_melspec()
