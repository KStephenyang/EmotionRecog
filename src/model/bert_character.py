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
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
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
        # pooled_output = sequence_output[:, 0]
        # embed_script = self.embed_script(script_id_id)
        # embed_script = script_id_id.view(embed_text.shape[0], -1)
        # # embed_scene = self.embed_scene(scene_num_id)
        # embed_scene = scene_num_id.view(embed_text.shape[0], -1)
        # # embed_sentence = self.embed_sentence(sentence_num_id)
        # embed_sentence = sentence_num_id.view(embed_text.shape[0], -1)
        # # embed_character = self.embed_character(character_id)
        # embed_character = character_id.view(embed_text.shape[0], -1)
        # embeddings = torch.cat(
        #     (embeddings, dense_features),
        #     dim=1
        # )
        # 生成输出结果
        # output = self.ffd0(pooled_output)
        return ModelOutput(
            love=love, joy=joy, fright=fright,
            anger=anger, fear=fear, sorrow=sorrow
        )
