from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any

from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorContextEmotion:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]):
        pre_pre_inputs, pre_inputs, inputs = list(), list(), list()
        for feature in features:
            inputs_dict = dict()
            inputs_dict["input_ids"] = feature["input_ids"]
            inputs_dict["token_type_ids"] = feature["token_type_ids"]
            inputs_dict["attention_mask"] = feature["attention_mask"]
            if "label" in feature:
                inputs_dict["label"] = feature["label"]
            inputs.append(inputs_dict)

            pre_inputs_dict = dict()
            pre_inputs_dict["input_ids"] = feature["pre_input_ids"]
            pre_inputs_dict["token_type_ids"] = feature["pre_token_type_ids"]
            pre_inputs_dict["attention_mask"] = feature["pre_attention_mask"]
            pre_inputs.append(pre_inputs_dict)

            pre_pre_inputs_dict = dict()
            pre_pre_inputs_dict["input_ids"] = feature["pre_pre_input_ids"]
            pre_pre_inputs_dict["token_type_ids"] = feature["pre_pre_token_type_ids"]
            pre_pre_inputs_dict["attention_mask"] = feature["pre_pre_attention_mask"]
            pre_pre_inputs.append(pre_pre_inputs_dict)

        batch = self.tokenizer.pad(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        pre_batch = self.tokenizer.pad(
            pre_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        pre_pre_batch = self.tokenizer.pad(
            pre_pre_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["pre_input_ids"] = pre_batch["input_ids"]
        batch["pre_token_type_ids"] = pre_batch["token_type_ids"]
        batch["pre_attention_mask"] = pre_batch["attention_mask"]

        batch["pre_pre_input_ids"] = pre_pre_batch["input_ids"]
        batch["pre_pre_token_type_ids"] = pre_pre_batch["token_type_ids"]
        batch["pre_pre_attention_mask"] = pre_pre_batch["attention_mask"]
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class DataCollatorEmotion:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]):
        inputs = list()
        for feature in features:
            inputs_dict = dict()
            inputs_dict["input_ids"] = feature["input_ids"]
            inputs_dict["token_type_ids"] = feature["token_type_ids"]
            inputs_dict["attention_mask"] = feature["attention_mask"]
            if "label" in feature:
                inputs_dict["label"] = feature["label"]
            inputs.append(inputs_dict)

        batch = self.tokenizer.pad(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


def data_collator_emotion(tokenizer, padding=True, max_length=512, model_type="bert_character"):
    if model_type == "bert_character_context":
        data_collator = DataCollatorContextEmotion(
            tokenizer=tokenizer, padding=padding, max_length=max_length
        )
    elif model_type == "bert_character":
        data_collator = DataCollatorEmotion(
            tokenizer=tokenizer, padding=padding, max_length=max_length
        )
    else:
        data_collator = DataCollatorEmotion(
            tokenizer=tokenizer, padding=padding, max_length=max_length
        )
    return data_collator
