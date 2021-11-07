import numpy as np
import pandas as pd
from config import train_config as const
from sklearn.metrics import classification_report


def eval_analysis(preds, labels):
    labels_name = ["love", "joy", "fright", "anger", "fear", "sorrow"]
    eval_info = list()
    recall_info = dict()
    for i in range(6):
        report = classification_report(
            preds[:, i], labels[:, i], labels=[0, 1, 2, 3],
            target_names=["no", "week", "medium", "strong"],
            output_dict=True
        )
        recall_info[labels_name[i]] = {
            "no": report["no"]["recall"],
            "week": report["week"]["recall"],
            "medium": report["medium"]["recall"],
            "strong": report["strong"]["recall"],
        }
        report = pd.DataFrame.from_dict(report)
        report.reset_index(inplace=True)
        report.insert(0, "label_name", labels_name[i])
        eval_info.append(report)
    eval_info = pd.concat(eval_info, axis=0, ignore_index=True)

    if const.TrainEnd:
        eval_info.to_csv(const.EvalAccCsv, index=False)
    return recall_info


def compute_metrics_bce(eval_preds):
    logits, labels = eval_preds
    e0 = np.argmax(logits[:, :4], axis=1)
    e1 = np.argmax(logits[:, 4:8], axis=1)
    e2 = np.argmax(logits[:, 8:12], axis=1)
    e3 = np.argmax(logits[:, 12:16], axis=1)
    e4 = np.argmax(logits[:, 16:20], axis=1)
    e5 = np.argmax(logits[:, 20:24], axis=1)
    preds = np.concatenate(
        (e0.reshape(-1, 1), e1.reshape(-1, 1), e2.reshape(-1, 1),
         e3.reshape(-1, 1), e4.reshape(-1, 1), e5.reshape(-1, 1)),
        axis=1
    )

    l0 = np.argmax(labels[:, :4], axis=1)
    l1 = np.argmax(labels[:, 4:8], axis=1)
    l2 = np.argmax(labels[:, 8:12], axis=1)
    l3 = np.argmax(labels[:, 12:16], axis=1)
    l4 = np.argmax(labels[:, 16:20], axis=1)
    l5 = np.argmax(labels[:, 20:24], axis=1)
    labels = np.concatenate(
        (l0.reshape(-1, 1), l1.reshape(-1, 1), l2.reshape(-1, 1),
         l3.reshape(-1, 1), l4.reshape(-1, 1), l5.reshape(-1, 1)),
        axis=1
    )
    acc = eval_analysis(preds, labels)
    score = 1 / (1 + np.mean((preds - labels) ** 2))
    return {"rmse ": score, "recall": acc}


def compute_metrics_ce(eval_preds):
    logits, labels = eval_preds
    e0 = np.argmax(logits[0], axis=1)
    e1 = np.argmax(logits[1], axis=1)
    e2 = np.argmax(logits[2], axis=1)
    e3 = np.argmax(logits[3], axis=1)
    e4 = np.argmax(logits[4], axis=1)
    e5 = np.argmax(logits[5], axis=1)
    preds = np.concatenate(
        (e0.reshape(-1, 1), e1.reshape(-1, 1), e2.reshape(-1, 1),
         e3.reshape(-1, 1), e4.reshape(-1, 1), e5.reshape(-1, 1)),
        axis=1
    )
    acc = eval_analysis(preds, labels)
    score = 1 / (1 + np.mean((preds - labels) ** 2))
    return {"rmse ": score, "recall": acc}
