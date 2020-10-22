# This file contains all useful paths
import os

PATH ="../data/"

# General directories
path_to_SONYCUST = os.path.join(PATH, "SONYC-UST")
path_to_ESC50 = os.path.join(PATH, "ESC-50")
path_to_UrbanSound8K = os.path.join(PATH, "UrbanSound8K")
path_to_Audioset = os.path.join(PATH, "Audioset")
path_to_summaries = os.path.join(PATH, "summaries")

# Pretrained weights
audioset = os.path.join(path_to_SONYCUST, "model/TALNet.pt")
audiosetCNN = os.path.join(path_to_SONYCUST, "model/Cnn14_mAP=0.431.pth")
audiosetCNN10 = os.path.join(path_to_SONYCUST, "model/Cnn10_mAP=0.380.pth")

# General things for SONYC-UST
path_to_annotation = os.path.join(path_to_SONYCUST, "annotations.csv")
path_to_taxonomy = os.path.join(path_to_SONYCUST, "dcase-ust-taxonomy.yaml")
wav_dir = os.path.join(path_to_SONYCUST, "audio")

# DCASE Baseline
emb_dir = os.path.join(path_to_SONYCUST, "embedding")

# TALNet
mel_dir = os.path.join(path_to_SONYCUST, "melTALNet")
