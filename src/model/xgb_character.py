from torch import nn
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.get("bert_path"))

    def forward(
            self,
            input_ids, token_type_ids, attention_mask
    ):
        _, pooled_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        return ModelOutput(embedding=pooled_output)
