from torch import nn
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertModel


class EmotionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.get("hidden_size"),
            config.get("d_out"))

    def forward(self, embeddings):
        pooled_output = self.dense(embeddings)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertCharacter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.get("bert_path"))
        self.out_love = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_joy = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_fright = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_anger = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_fear = nn.Linear(config.get("hidden_size"), config.get("d_out"))
        self.out_sorrow = nn.Linear(config.get("hidden_size"), config.get("d_out"))

    def forward(
            self,
            input_ids, token_type_ids, attention_mask
    ):
        # 使用Bert编码
        _, pooled_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        love = self.out_love(pooled_output)
        joy = self.out_joy(pooled_output)
        fright = self.out_fright(pooled_output)
        anger = self.out_anger(pooled_output)
        fear = self.out_fear(pooled_output)
        sorrow = self.out_sorrow(pooled_output)
        return ModelOutput(
            love=love, joy=joy, fright=fright,
            anger=anger, fear=fear, sorrow=sorrow
        )
