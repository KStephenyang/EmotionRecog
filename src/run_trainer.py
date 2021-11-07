import logging
import os

from datasets import load_from_disk, Dataset, DatasetDict
from transformers import BertTokenizer
from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from config import train_config as const
from dataset.collator import data_collator_emotion
from dataset.loader import EmotionDataset
from dataset.token_map import tokenize_map
from eval.metric import compute_metrics_ce
from eval.submit import generate_submit_result
from model import model_builder
from train.trainer import EmotionTrainer

logger = logging.getLogger(__name__)


def train():
    tokenizer = BertTokenizer.from_pretrained(const.BertPath)

    # 1.训练数据预处理
    if os.path.exists(const.DataPklPath):
        tokenized_datasets = load_from_disk(const.DataPklPath)
    else:
        emo_dataset = EmotionDataset(const.TrainDataPath)
        data_raw = emo_dataset.preprocess()
        train_df, valid_df = emo_dataset.split_train_valid(data_raw, seed=const.RandomSeed, num=30000)
        train_datasets = Dataset.from_pandas(train_df)
        valid_datasets = Dataset.from_pandas(valid_df)
        raw_datasets = DatasetDict()
        raw_datasets["train"] = train_datasets
        raw_datasets["validation"] = valid_datasets
        tokenized_datasets = raw_datasets.map(
            tokenize_map, batched=True,
            fn_kwargs={"tokenizer": tokenizer, "model_type": const.ModelType}
        )
        tokenized_datasets.save_to_disk(const.DataPklPath)

    data_collator = data_collator_emotion(
        tokenizer=tokenizer, padding=True, max_length=512, model_type=const.ModelType
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

    model = model_builder(const.BertPath, tokenizer, const.ModelType)
    # 3.构建训练器
    trainer = EmotionTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"].select(range(0, 20)),
        eval_dataset=tokenized_datasets["validation"].select(range(0, 8)),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_ce
    )
    # 4.模型训练
    trainer.train()
    const.TrainEnd = True
    # 5.生成模型评估结果
    trainer.evaluate()
    # 6.生成提交结果

    test_dataset = EmotionDataset(const.TestDataPath)
    test_data = test_dataset.preprocess()
    test_datasets = Dataset.from_pandas(test_data)
    test_datasets = test_datasets.map(
        tokenize_map, batched=True, num_proc=4,
        fn_kwargs={"tokenizer": tokenizer, "model_type": const.ModelType}
    )
    test_result = trainer.predict(test_datasets)
    generate_submit_result(
        test_data, test_result, const.ModelFusionPkl, const.SubmitFileTsv
    )


if __name__ == "__main__":
    train()
