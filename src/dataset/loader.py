import pandas as pd


class EmotionDataset:
    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def get_script_scene(data):
        data["script"] = data["id"].str.split("_").str[0].astype(int)
        data["scene"] = data["id"].str.split("_").str[1].astype(int)
        data["sent_id"] = data["id"].str.split("_").str[3].astype(int)
        data["script_scene"] = (
                data["id"].str.split("_").str[0] + "_" +
                data["id"].str.split("_").str[1] + "_" +
                data["id"].str.split("_").str[2]
        )
        data.sort_values(by=["script", "scene", "sent_id"], inplace=True)
        data.reset_index(inplace=True, drop=True)
        return data

    @staticmethod
    def get_content_map(data):
        script_scene_content_dict = dict()
        script_scene_id_dict = dict()
        for script_scene, item in data.groupby(by=["script_scene"]):
            sent_list = item["content"].unique().tolist()
            sent_id_dict = {sent: i for i, sent in enumerate(sent_list)}
            script_scene_content_dict[script_scene] = sent_list
            script_scene_id_dict[script_scene] = sent_id_dict
        return script_scene_content_dict, script_scene_id_dict

    def get_content_pre(self, data):
        index_list = list()
        pre_content_list = list()
        pre_pre_content_list = list()
        script_scene_content_dict, script_scene_id_dict = self.get_content_map(data)
        for script_scene, group_item in data.groupby(by=["script_scene"]):
            sent_list = script_scene_content_dict[script_scene]
            sent_id_dict = script_scene_id_dict[script_scene]
            for i in group_item.index:
                content = group_item.loc[i, "content"]

                sent_unid = sent_id_dict[content]

                if sent_unid < 1:
                    pre_pre_content = "[START]"
                    pre_content = "[START]"
                elif sent_unid < 2:
                    pre_pre_content = "[START]"
                    pre_content = sent_list[sent_unid - 1]
                else:
                    pre_pre_content = sent_list[sent_unid - 2]
                    pre_content = sent_list[sent_unid - 1]

                index_list.append(i)
                pre_content_list.append(pre_content)
                pre_pre_content_list.append(pre_pre_content)
        data.loc[index_list, "pre_content"] = pre_content_list
        data.loc[index_list, "pre_pre_content"] = pre_pre_content_list
        return data

    def preprocess(self):
        data = pd.read_csv(self.data_path, sep='\t')
        data = self.get_script_scene(data)
        data = self.get_content_pre(data)
        del data["script"]
        del data["scene"]
        del data["sent_id"]
        return data

    @staticmethod
    def split_train_valid(data, seed=10, num=30000):
        train_raw = data[~data["emotions"].isna()].reset_index(drop=True)
        indexes = train_raw.index.tolist()
        import random
        random.seed(seed)
        random.shuffle(indexes)
        train_data = train_raw.loc[indexes[:num]].reset_index(drop=True)
        valid_data = train_raw.loc[indexes[num:]].reset_index(drop=True)
        return train_data, valid_data
