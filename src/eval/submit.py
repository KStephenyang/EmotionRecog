from datasets import Dataset
from torch import nn
import numpy as np
import torch

from dataset.loader import EmotionDataset


def generate_submit_result(test_data, test_result, model_fusion_path, submit_file_path):

    out_sf = nn.Softmax()
    e0 = np.argmax(test_result.predictions[0], axis=1)
    e0_prob = out_sf(torch.tensor(test_result.predictions[0]))

    e1 = np.argmax(test_result.predictions[1], axis=1)
    e1_prob = out_sf(torch.tensor(test_result.predictions[1]))

    e2 = np.argmax(test_result.predictions[2], axis=1)
    e2_prob = out_sf(torch.tensor(test_result.predictions[2]))

    e3 = np.argmax(test_result.predictions[3], axis=1)
    e3_prob = out_sf(torch.tensor(test_result.predictions[3]))

    e4 = np.argmax(test_result.predictions[4], axis=1)
    e4_prob = out_sf(torch.tensor(test_result.predictions[4]))

    e5 = np.argmax(test_result.predictions[5], axis=1)
    e5_prob = out_sf(torch.tensor(test_result.predictions[5]))

    emotions = []
    for i in range(e0.shape[0]):
        emotion = ','.join(
            [str(e0[i]), str(e1[i]), str(e2[i]), str(e3[i]), str(e4[i]), str(e5[i])]
        )
        emotions.append(emotion)
    test_data['emotion'] = emotions

    test_data['love'] = e0_prob.tolist()
    test_data['joy'] = e1_prob.tolist()
    test_data['fright'] = e2_prob.tolist()
    test_data['anger'] = e3_prob.tolist()
    test_data['fear'] = e4_prob.tolist()
    test_data['sorrow'] = e5_prob.tolist()

    test_data[['id', 'emotion']].to_csv(submit_file_path, sep='\t', index=False)
    test_data[
        [
            'id', 'content', 'emotion',
            "love", "joy", "fright", "anger", "fear", "sorrow"
        ]
    ].to_pickle(model_fusion_path)
