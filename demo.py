# """
# Author: Jeffrey Luo, Monta Vista High School, Cupertino, CA
# Date Created: 09/2023
# Copyright (c) 2023 Jeff Luo
# License: MIT
# """
# import torch
# import numpy as np
# import pandas as pd

# from model_torch import Audio2EmotionModel

# def rescale(scores:np.ndarray) -> np.ndarray:
#     """
#     Rescale the model output back into original range (1 ~ 7.83)
    
#     Args:
#         scores: the model prediction in [0,1], shape (B, 8)

#     Returns:
#         Rescaled model prediction in [1, 7.83], shape (B, 8)
#     """
#     scores = scores * 10
#     for r in range(scores.shape[0]):
#         for s in range(scores.shape[-1]):
#             if scores[r,s] < 1:
#                 scores[r,s] = 1
#             elif scores[r,s] > 7.83:
#                 scores[r,s] = 7.83
    
#     return scores

# def print_scores(scores: np.ndarray) -> None:
#     """
#     Pretty print for print out scores 
#     """
#     df = pd.DataFrame(scores, columns=['valence', 'energy', 'tension', 
#                                        'anger', 'fear', 'happy', 'sad', 'tender'])
#     print(df)


# if __name__ == '__main__':
#     # config
#     checkpoint_path = 'MusicEmotionDetection/weights/best.pth'
#     # test audio spectrograms
#     audio_spectrograms_path = 'MusicEmotionDetection/data/spectrograms/345.mp3.npy'

#     # init model and loading weights
#     model = Audio2EmotionModel()
#     model.load_state_dict(torch.load(checkpoint_path))
#     model.eval()
#     model.cuda()

#     # load test audio spectrograms
#     audio_spectrograms = np.load(audio_spectrograms_path)
#     # add batch,chanel dim
#     audio_spectrograms = np.expand_dims(audio_spectrograms, (0, 1))

#     audio_spectrograms = torch.tensor(audio_spectrograms)

#     # inference
#     with torch.no_grad():
#         audio_spectrograms = audio_spectrograms.to('cuda')
#         pred = model(audio_spectrograms)
#         # offload to cpu
#         pred = pred.cpu().numpy()

#         # shape check: if single batch add batch dim
#         if len(pred.shape) == 1:
#             pred = np.expand_dims(pred, 0)

#         # rescale
#         pred_rescaled = rescale(pred)
#         # print result
#         print_scores(pred_rescaled)


"""
Author: Jeffrey Luo, Monta Vista High School, Cupertino, CA
Date Created: 09/2023
Copyright (c) 2023 Jeff Luo
License: MIT
"""
import torch
import numpy as np
import pandas as pd

from model_torch import Audio2EmotionModel

def rescale(scores: np.ndarray) -> np.ndarray:
    """
    Rescale the model output back into original range (1 ~ 7.83)
    
    Args:
        scores: the model prediction in [0,1], shape (B, 8)

    Returns:
        Rescaled model prediction in [1, 7.83], shape (B, 8)
    """
    scores = scores * 10
    for r in range(scores.shape[0]):
        for s in range(scores.shape[-1]):
            if scores[r, s] < 1:
                scores[r, s] = 1
            elif scores[r, s] > 7.83:
                scores[r, s] = 7.83
    
    return scores

def print_scores(scores: np.ndarray) -> None:
    """
    Pretty print for print out scores 
    """
    df = pd.DataFrame(scores, columns=['valence', 'energy', 'tension', 
                                       'anger', 'fear', 'happy', 'sad', 'tender'])
    print(df)

if __name__ == '__main__':
    # config
    checkpoint_path = '/home/nithish/Desktop/Tonal_Based_Emotion_Synthesis/weights/best.pth'
    # test audio spectrograms
    audio_spectrograms_path = '/home/nithish/Desktop/Tonal_Based_Emotion_Synthesis/data/spectrograms/201.mp3.npy'

    # init model and loading weights
    model = Audio2EmotionModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))  # Load model on CPU
    model.eval()

    # load test audio spectrograms
    audio_spectrograms = np.load(audio_spectrograms_path)
    # add batch, channel dim
    audio_spectrograms = np.expand_dims(audio_spectrograms, (0, 1))

    audio_spectrograms = torch.tensor(audio_spectrograms)

    # inference
    with torch.no_grad():
        # No need to move to GPU, keep everything on CPU
        pred = model(audio_spectrograms)
        # offload to numpy
        pred = pred.numpy()

        # shape check: if single batch add batch dim
        if len(pred.shape) == 1:
            pred = np.expand_dims(pred, 0)

        # rescale
        pred_rescaled = rescale(pred)
        # print result
        print_scores(pred_rescaled)
