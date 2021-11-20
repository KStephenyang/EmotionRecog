import joblib
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import torch
from config import train_config as const
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict


def lgb_model_fit_sklearn(data, mode="love"):
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=4, random_state=const.RandomSeed,
        reg_lambda=0.2, reg_alpha=0.1, class_weight="balanced", silent=False
    )
    result = cross_validate(
        model, torch.tensor(data["embedding"]), data[mode],
        cv=5, return_estimator=True, n_jobs=4
    )
    print(f"{mode} mode score:", result["test_score"])
    print(f"{mode} mode score mean:", result["test_score"].mean())
    joblib.dump(result["estimator"], f'../output/lgb_{mode}_cv.pkl')


def lgb_model_fit(data, mode="love"):
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 6,
        'metric': 'f1 macro',
        'objective': 'multiclass',
        'subsample': 0.8,
        'num_class': 4,
        'num_threads': 5
    }
    result = lgb.cv(
        params,
        train_set=lgb.Dataset(torch.tensor(data["embedding"]), label=data[mode]),
        nfold=5, return_cvbooster=True
    )
    print(f"{mode} mode score:", result["f1 macro-mean"])
    joblib.dump(result["cvbooster"], f'../output/lgb_{mode}_cv.pkl')


def lightgbm_train_cv():
    train_dt = pd.read_pickle(const.TrainDataEmbedPath)

    train_dt = train_dt[~train_dt["emotions"].isna()].reset_index(drop=True)
    train_dt["love"] = train_dt["emotions"].str.split(",").str[0].astype(int)
    train_dt["joy"] = train_dt["emotions"].str.split(",").str[1].astype(int)
    train_dt["fright"] = train_dt["emotions"].str.split(",").str[2].astype(int)
    train_dt["anger"] = train_dt["emotions"].str.split(",").str[3].astype(int)
    train_dt["fear"] = train_dt["emotions"].str.split(",").str[4].astype(int)
    train_dt["sorrow"] = train_dt["emotions"].str.split(",").str[5].astype(int)
    # 准备数据
    lgb_model_fit_sklearn(train_dt, "love")
    lgb_model_fit_sklearn(train_dt, "joy")
    lgb_model_fit_sklearn(train_dt, "fright")
    lgb_model_fit_sklearn(train_dt, "anger")
    lgb_model_fit_sklearn(train_dt, "fear")
    lgb_model_fit_sklearn(train_dt, "sorrow")


def lightgbm_train():
    train_dt = pd.read_pickle(const.TrainDataEmbedPath)

    train_dt = train_dt[~train_dt["emotions"].isna()].reset_index(drop=True)
    train_dt["love"] = train_dt["emotions"].str.split(",").str[0].astype(int)
    train_dt["joy"] = train_dt["emotions"].str.split(",").str[1].astype(int)
    train_dt["fright"] = train_dt["emotions"].str.split(",").str[2].astype(int)
    train_dt["anger"] = train_dt["emotions"].str.split(",").str[3].astype(int)
    train_dt["fear"] = train_dt["emotions"].str.split(",").str[4].astype(int)
    train_dt["sorrow"] = train_dt["emotions"].str.split(",").str[5].astype(int)

    # 准备数据
    X = torch.tensor(train_dt["embedding"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, train_dt[["love", "joy", "fright", "anger", "fear", "sorrow"]],
        test_size=0.2, random_state=const.RandomSeed
    )
    # 训练
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 6,
        'objective': 'multiclass',
        'num_class': 4,
    }
    clf_model = dict()
    clf_love = lgb.train(
        params, lgb.Dataset(X_train, label=y_train["love"]),
        valid_sets=[lgb.Dataset(X_test, label=y_test["love"])]
    )
    clf_model["love"] = clf_love

    clf_joy = lgb.train(
        params, lgb.Dataset(X_train, label=y_train["joy"]),
        valid_sets=[lgb.Dataset(X_test, label=y_test["joy"])]
    )
    clf_model["joy"] = clf_joy

    clf_fright = lgb.train(
        params, lgb.Dataset(X_train, label=y_train["fright"]),
        valid_sets=[lgb.Dataset(X_test, label=y_test["fright"])]
    )
    clf_model["fright"] = clf_fright

    clf_anger = lgb.train(
        params, lgb.Dataset(X_train, label=y_train["anger"]),
        valid_sets=[lgb.Dataset(X_test, label=y_test["anger"])]
    )
    clf_model["anger"] = clf_anger

    clf_fear = lgb.train(
        params, lgb.Dataset(X_train, label=y_train["fear"]),
        valid_sets=[lgb.Dataset(X_test, label=y_test["fear"])]
    )
    clf_model["fear"] = clf_fear

    clf_sorrow = lgb.train(
        params, lgb.Dataset(X_train, label=y_train["sorrow"]),
        valid_sets=[lgb.Dataset(X_test, label=y_test["sorrow"])]
    )
    clf_model["sorrow"] = clf_sorrow

    acc_recall_eval(clf_model, X_test, y_test, "love")
    acc_recall_eval(clf_model, X_test, y_test, "joy")
    acc_recall_eval(clf_model, X_test, y_test, "fright")
    acc_recall_eval(clf_model, X_test, y_test, "anger")
    acc_recall_eval(clf_model, X_test, y_test, "fear")
    acc_recall_eval(clf_model, X_test, y_test, "sorrow")


def acc_recall_eval(clf_model, x_test, y_test, mode="love"):
    # 1、AUC
    clf = clf_model[mode]
    y_pred_pa = clf.predict(x_test)  # !!!注意lgm预测的是分数，类似 sklearn的predict_proba
    y_test_oh = label_binarize(y_test[mode], classes=[0, 1, 2, 3])
    print(F'情感{mode}的auc：', roc_auc_score(y_test_oh, y_pred_pa, average='micro'))
    # 2、模型报告
    y_pred = y_pred_pa.argmax(axis=1)
    print(classification_report(y_test[mode], y_pred))
    joblib.dump(clf, f'../output/lgb_{mode}.pkl')


def lgb_cv_predict(lgb_models, x_test):
    pred_prob = 0
    for model in lgb_models:
        pred_prob += model.predict_proba(x_test)
    pred_prob = pred_prob / len(lgb_models)
    return pred_prob.argmax(axis=1)


def x_test_predict(mode="_cv"):
    love_lgb = joblib.load(f'../output/lgb_love{mode}.pkl')
    joy_lgb = joblib.load(f'../output/lgb_joy{mode}.pkl')
    fright_lgb = joblib.load(f'../output/lgb_fright{mode}.pkl')
    anger_lgb = joblib.load(f'../output/lgb_anger{mode}.pkl')
    fear_lgb = joblib.load(f'../output/lgb_fear{mode}.pkl')
    sorrow_lgb = joblib.load(f'../output/lgb_sorrow{mode}.pkl')
    test_dt = pd.read_pickle(const.TestDataEmbedPath)
    x_test_embed = torch.tensor(test_dt["embedding"])

    love = lgb_cv_predict(love_lgb, x_test_embed)
    joy = lgb_cv_predict(joy_lgb, x_test_embed)
    fright = lgb_cv_predict(fright_lgb, x_test_embed)
    anger = lgb_cv_predict(anger_lgb, x_test_embed)
    fear = lgb_cv_predict(fear_lgb, x_test_embed)
    sorrow = lgb_cv_predict(sorrow_lgb, x_test_embed)

    emotions = []
    for i in range(love.shape[0]):
        emotion = ','.join(
            [str(love[i]), str(joy[i]), str(fright[i]), str(anger[i]), str(fear[i]), str(sorrow[i])]
        )
        emotions.append(emotion)
    test_dt['emotion'] = emotions

    test_dt[['id', 'emotion']].to_csv(const.SubmitFileTsv, sep='\t', index=False)

    # test_dt["love"] = love_prob
    # test_dt["joy"] = joy_prob
    # test_dt["fright"] = fright_prob
    # test_dt["anger"] = anger_prob
    # test_dt["fear"] = fear_prob
    # test_dt["sorrow"] = sorrow_prob
    # test_dt[
    #     [
    #         'id', 'content', 'emotion',
    #         "love", "joy", "fright", "anger", "fear", "sorrow"
    #     ]
    # ].to_pickle(const.ModelFusionPkl)


# lightgbm_train_cv()
x_test_predict()
