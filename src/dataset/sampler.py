import random
from tqdm import tqdm
import torch


class EmotionSampler:
    def __init__(self, dataset, batch_size, drop_last=False, padding_idx=0):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.padding_idx = padding_idx
        self.dataset = self.sample(dataset)

    def sample(self, dataset):
        count = 0
        sample_batch = list()
        src, tgt = list(), list()
        random.shuffle(dataset.examples)
        for example in tqdm(dataset.examples, desc="Sample Batch Data"):
            if count == self.batch_size:
                src = torch.tensor(self.__pad(src, self.padding_idx), dtype=torch.long)
                tgt = torch.tensor(self.__pad(tgt, self.padding_idx), dtype=torch.long)
                batch = {"src": src, "tgt": tgt}
                sample_batch.append(batch)
                src, tgt = list(), list()
                count = 0
            src.append(example["src"])
            tgt.append(example["tgt"])
            count += 1
        if not self.drop_last:
            batch = {"src": src, "tgt": tgt}
            sample_batch.append(batch)
        return sample_batch

    @staticmethod
    def __pad(data, sig, width=512):
        max_len = max(len(d) for d in data)
        if max_len < width:
            pad_data = [d + [sig] * (max_len - len(d)) for d in data]
        else:
            pad_data = list()
            for d in data:
                if len(d) < width:
                    pad_data.append(d + [sig] * (width - len(d)))
                else:
                    pad_data.append(d[:width])
        return pad_data

    def __iter__(self):
        yield from self.dataset

    def __len__(self):
        return len(self.dataset)
