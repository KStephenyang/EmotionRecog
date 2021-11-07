import os
from datetime import datetime
import torch

ProPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TrainDataPath = os.path.join(ProPath, "data/train_dataset_v2.tsv")
TestDataPath = os.path.join(ProPath, "data/test_dataset.tsv")

TrainDataPathCSV = os.path.join(ProPath, "data/train_dataset_scene_v1_train.csv")
ValidDataPathCSV = os.path.join(ProPath, "data/train_dataset_scene_v1_valid.csv")
TestDataPathCSV = os.path.join(ProPath, "data/test_dataset.tsv")

BertPath = os.path.join(ProPath, "pretrained/chinese-roberta-wwm-ext")
ModelSavePath = os.path.join(ProPath, "output/emotion_simple_model_{}.pt")

if torch.cuda.is_available():
    Device = "cuda"
else:
    Device = "cpu"

OutputPath = os.path.join(ProPath, "output")
LogPath = os.path.join(ProPath, "log")
DataPklPath = os.path.join(ProPath, "data/dataset_train")

RandomSeed = 10
ModelType = "bert_character_context"
suffix = datetime.now().strftime("%Y%m%d%H%M")
TestResult = os.path.join(ProPath, "output/test_result.pkl")
SubmitFileTsv = os.path.join(ProPath, f"output/submit_{ModelType}_{suffix}.tsv")
EvalInfoCsv = os.path.join(ProPath, f"output/eval_info_{ModelType}_{suffix}.csv")
EvalAccCsv = os.path.join(ProPath, f"output/eval_acc_{ModelType}_{suffix}.csv")
ModelFusionPkl = os.path.join(ProPath, f"output/model_fusion_{ModelType}_{suffix}.pkl")
TrainEnd = False
