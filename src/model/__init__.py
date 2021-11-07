from .bert_context import BertCharacterContext
from .bert_character import BertCharacter
from .xgb_character import *


def model_builder(bert_path, tokenizer, model_type):
    config = {
        "padding_idx": tokenizer.pad_token_id,
        "bert_path": bert_path,
        "hidden_size": 768,
        "d_out": 4,
        "embed_script_dim_in": 49,
        "embed_script_dim_out": 10,
        "embed_scene_dim_in": 180,
        "embed_scene_dim_out": 30,
        "embed_sentence_dim_in": 2339,
        "embed_sentence_dim_out": 100,
        "embed_character_dim_in": 74,
        "embed_character_dim_out": 20
    }

    if model_type == "bert_character_context":
        model = BertCharacterContext(config)
    elif model_type == "bert_character":
        model = BertCharacter(config)
    else:
        model = BertCharacter(config)
    return model
