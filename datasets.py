"""
Author: Jeffrey Luo, Monta Vista High School, Cupertino, CA
Date Created: 09/2023
Copyright (c) 2023 Jeff Luo
License: MIT
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader



class AudioEmotionDataset(Dataset):
    """
    Dataset class for Audio to emotion model

    Args:
        data_list: A list of file names for this dataset (filename ONLY)
        data_path: spectrograms files folder path
        anno_path: annotation files path (full path)

    """
    def __init__(self, data_list:list, data_path:str, anno_path:str) -> None:
        super().__init__()
        self.data_list = data_list
        self.data_path = data_path
        self.anno_path = anno_path

        self.audio_spec = []
        self.emotion_anno = []

        self._load_data()
    
    def _load_data(self):
        """
        Helper function, load data into memory according to data list
        """
        # load annotation
        annos = pd.read_csv(self.anno_path, index_col=0)
        # drop last col
        annos = annos.drop(columns=['TARGET'])
        # scale by 0.1 (as paper)
        annos = annos * 0.1
        
        print('Loading dataset...')
        for sample in self.data_list:
            spectrogram = np.load(os.path.join(self.data_path, f'{sample}.mp3.npy'))
            # add chanel dim
            spectrogram = np.expand_dims(spectrogram, 0)
            sample_anno = annos.iloc[int(sample)-1].to_numpy(dtype=np.float32)
            self.audio_spec.append(spectrogram)
            self.emotion_anno.append(sample_anno)

    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        Get item will return tuples of audio spectrograms(np array, float32) and
        emotion annotation of this sample(np array, float32)
        """
        return self.audio_spec[index], self.emotion_anno[index]


def build_default_dataloader(data_list:list) -> DataLoader:
    """
    Get data loader for training with default setting from paper

    Args: 
        data_list: a list of data names for dataset init

    Returns:
        A dataloader class
    """

    dataset = AudioEmotionDataset(data_list, 'MusicEmotionDetection/data/spectrograms', 'MusicEmotionDetection/data/mean_ratings_set1.csv')
    return DataLoader(dataset, 8, shuffle=True)