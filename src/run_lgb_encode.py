import logging
import os

import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import BertTokenizer, Trainer
from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from config import train_config as const
from dataset.collator import data_collator_emotion
from dataset.token_map import tokenize_map
from model import model_builder

logger = logging.getLogger(__name__)


def train():
    tokenizer = BertTokenizer.from_pretrained(const.BertPath)

    # 1.训练数据预处理
    if os.path.exists(const.EncodeDatasetPath):
        raw_datasets = load_from_disk(const.EncodeDatasetPath)
    else:
        # train_dataset = EmotionXgbDataset(const.TrainDataPath)
        # train_dataset.preprocess()
        # train_data = train_dataset.get_data()
        train_data = pd.read_csv(const.TrainDataPath, sep='\t')
        del train_data["emotions"]
        train_datasets = Dataset.from_pandas(train_data)
        # test_dataset = EmotionXgbDataset(const.TestDataPath)
        # test_dataset.preprocess()
        # test_data = test_dataset.get_data()
        test_data = pd.read_csv(const.TestDataPath, sep='\t')
        test_datasets = Dataset.from_pandas(test_data)
        raw_datasets = DatasetDict()

        raw_datasets["train"] = train_datasets
        raw_datasets["test"] = test_datasets
        raw_datasets = raw_datasets.map(
            tokenize_map, batched=True,
            fn_kwargs={"tokenizer": tokenizer, "model_type": const.EncodeType}
        )
        raw_datasets.save_to_disk(const.EncodeDatasetPath)

    data_collator = data_collator_emotion(
        tokenizer=tokenizer, padding=True, max_length=512, model_type=const.EncodeType
    )

    # 2.构建模型
    training_args = TrainingArguments(
        output_dir=const.OutputPath,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        # gradient_accumulation_steps=2,
        logging_dir=const.LogPath,
        logging_steps=2,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        save_strategy=IntervalStrategy.EPOCH,
        evaluation_strategy=IntervalStrategy.EPOCH,
        eval_steps=2,
        save_steps=4,
        save_total_limit=3
    )

    model = model_builder(const.BertPath, tokenizer, const.EncodeType)
    # 3.构建训练器
    trainer = Trainer(
        model,
        training_args,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    # 4.embed训练数据
    train_embed = trainer.predict(raw_datasets["train"].select(range(0, 20)))
    train_data = pd.read_csv(const.TrainDataPath, sep='\t').iloc[0:20]
    train_data["embedding"] = train_embed.predictions.tolist()
    train_data.to_pickle(const.TrainDataEmbedPath)
    # 5.embed测试数据
    test_embed = trainer.predict(raw_datasets["test"].select(range(0, 20)))
    test_data = pd.read_csv(const.TestDataPath, sep='\t').iloc[0:20]
    test_data["embedding"] = test_embed.predictions.tolist()
    test_data.to_pickle(const.TestDataEmbedPath)


if __name__ == "__main__":
    train()
