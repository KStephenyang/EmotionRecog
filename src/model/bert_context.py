from torch import nn
import torch
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertModel


class EmotionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embed_dim = (config.get("embed_script_dim_out") + config.get("embed_scene_dim_out") + config.get(
        #     "embed_sentence_dim_out") + config.get("embed_character_dim_out"))
        # embed_dim = 4
        # dense_dim = config.get("dense_dim")
        self.dense = nn.Linear(
            # config.get("hidden_size") + embed_dim + dense_dim,
            config.get("hidden_size"),
            config.get("d_out"))
        # self.activation = nn.Sigmoid()

    def forward(self, embeddings):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(embeddings)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertCharacterContext(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.get("bert_path"))
        self.lstm = nn.LSTM(
            input_size=config.get("hidden_size"),
            hidden_size=config.get("hidden_size"),
            num_layers=4, batch_first=True
        )
        self.out_love = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_joy = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_fright = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_anger = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_fear = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_sorrow = nn.Linear(config.get("hidden_size"), config.get("d_out"))

    def forward(
            self,
            input_ids, token_type_ids, attention_mask,
            pre_input_ids, pre_token_type_ids, pre_attention_mask,
            pre_pre_input_ids, pre_pre_token_type_ids, pre_pre_attention_mask,
    ):
        _, pre_pre_pooled_output = self.encoder(
            input_ids=pre_pre_input_ids,
            token_type_ids=pre_pre_token_type_ids,
            attention_mask=pre_pre_attention_mask,
            return_dict=False
        )
        _, pre_pooled_output = self.encoder(
            input_ids=pre_input_ids,
            token_type_ids=pre_token_type_ids,
            attention_mask=pre_attention_mask,
            return_dict=False
        )
        # 使用Bert编码
        _, pooled_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        pooled_output = torch.cat((
            pre_pre_pooled_output.unsqueeze(1),
            pre_pooled_output.unsqueeze(1),
            pooled_output.unsqueeze(1)),
            dim=1
        )
        lstm_out, (hidden_last, cn_last) = self.lstm(pooled_output)
        output = hidden_last[-1]
        love = self.out_love(output)
        joy = self.out_joy(output)
        fright = self.out_fright(output)
        anger = self.out_anger(output)
        fear = self.out_fear(output)
        sorrow = self.out_sorrow(output)
        # 生成输出结果
        return ModelOutput(
            love=love, joy=joy, fright=fright,
            anger=anger, fear=fear, sorrow=sorrow
        )
