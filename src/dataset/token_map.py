class EmotionTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def token(self, example):
        inputs = self.tokenizer(example["content"], truncation=True)
        if "emotions" in example:
            labels = list()
            for emotion in example["emotions"]:
                label = emotion.split(",")
                label = [int(label) for label in label]
                labels.append(label)
            inputs["label"] = labels
        return inputs


class CharacterEmotionTokenizer(EmotionTokenizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def mask_character(self, character, inputs):
        for i, el in enumerate(character):
            if el is None:
                character[i] = "[UNK]"
        character_token = self.tokenizer(character)
        for i, char_token in enumerate(character_token["input_ids"]):
            if 100 in char_token:
                char_token.remove(100)
            if 101 in char_token:
                char_token.remove(101)
            if 102 in char_token:
                char_token.remove(102)
            for token in char_token:
                input_ids = inputs["input_ids"][i]
                for ind, el in enumerate(input_ids):
                    if el == token:
                        inputs["token_type_ids"][i][ind] = 1
        return inputs

    def token(self, example):
        inputs = self.tokenizer(example["content"], truncation=True)
        inputs = self.mask_character(example["character"], inputs)
        if "emotions" in example:
            labels = list()
            for emotion in example["emotions"]:
                label = emotion.split(",")
                label = [int(label) for label in label]
                labels.append(label)
            inputs["label"] = labels
        return inputs


class CharacterContextEmotionTokenizer(EmotionTokenizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def mask_character(self, character, inputs):
        for i, el in enumerate(character):
            if el is None:
                character[i] = "[UNK]"
        character_token = self.tokenizer(character)
        for i, char_token in enumerate(character_token["input_ids"]):
            if 100 in char_token:
                char_token.remove(100)
            if 101 in char_token:
                char_token.remove(101)
            if 102 in char_token:
                char_token.remove(102)
            for token in char_token:
                input_ids = inputs["input_ids"][i]
                for ind, el in enumerate(input_ids):
                    if el == token:
                        inputs["token_type_ids"][i][ind] = 1
        return inputs

    def token(self, example):
        inputs = self.tokenizer(example["content"], truncation=True)
        inputs = self.mask_character(example["character"], inputs)

        pre_inputs = self.tokenizer(example["pre_content"], truncation=True)
        pre_inputs["pre_input_ids"] = pre_inputs.pop("input_ids")
        pre_inputs["pre_token_type_ids"] = pre_inputs.pop("token_type_ids")
        pre_inputs["pre_attention_mask"] = pre_inputs.pop("attention_mask")

        pre_pre_inputs = self.tokenizer(example["pre_pre_content"], truncation=True)
        pre_pre_inputs["pre_pre_input_ids"] = pre_pre_inputs.pop("input_ids")
        pre_pre_inputs["pre_pre_token_type_ids"] = pre_pre_inputs.pop("token_type_ids")
        pre_pre_inputs["pre_pre_attention_mask"] = pre_pre_inputs.pop("attention_mask")
        if "emotions" in example:
            labels = list()
            for emotion in example["emotions"]:
                label = emotion.split(",")
                label = [int(label) for label in label]
                labels.append(label)
            inputs["label"] = labels
        inputs.update(pre_inputs)
        inputs.update(pre_pre_inputs)
        return inputs


def tokenize_map(example, tokenizer, model_type="bert_character"):
    if model_type == "bert_character":
        token_func = CharacterEmotionTokenizer(tokenizer)
    elif model_type == "bert_character_context":
        token_func = CharacterContextEmotionTokenizer(tokenizer)
    else:
        token_func = EmotionTokenizer(tokenizer)
    return token_func.token(example)
