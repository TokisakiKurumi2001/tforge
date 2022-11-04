from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple
from TForge import TForgeTokenizer

class TForgeDataLoader:
    def __init__(
        self, source_tokenizer_ck: str, target_tokenizer_ck: str, translate_tokenizer_ck: str,
        input_max_length: int, keyword_max_length: int, output_max_length: int, translate_max_length: int
        ):
        dataset = load_dataset('csv', data_files='data/data.csv')
        dataset = dataset['train'].train_test_split(test_size=0.3, seed=42)
        test_valid_dataset = dataset.pop('test')
        test_valid_dataset = test_valid_dataset.train_test_split(test_size=0.5, seed=42)
        dataset['valid'] = test_valid_dataset.pop('train')
        dataset['test'] = test_valid_dataset.pop('test')
        self.dataset = dataset
        self.tokenizer = TForgeTokenizer(source_tokenizer_ck, target_tokenizer_ck, translate_tokenizer_ck)
        self.input_max_length = input_max_length
        self.keyword_max_length = keyword_max_length
        self.output_max_length = output_max_length
        self.translate_max_length = translate_max_length

    def __shift_tokens_left(self, input_ids: torch.Tensor, pad_token_id: int):
        """
        Shift input ids one token to the left.
        """
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
        shifted_input_ids[:, -1] = pad_token_id
        return shifted_input_ids

    def __create_masks(self, input: Tensor, pad_token_id: int, is_enc: bool) -> Tensor:
        if is_enc:
            source_pad_mask = (input != pad_token_id).unsqueeze(1) # (batch_size, 1, max_target_sequence_length)
            return source_pad_mask
        else:
            target_pad_mask = (input != pad_token_id).unsqueeze(1) # (batch_size, 1, max_source_sequence_length)

            # subsequent mask for decoder inputs
            max_target_sequence_length = input.shape[1]
            target_attention_square = (max_target_sequence_length, max_target_sequence_length)

            full_mask = torch.full(target_attention_square, 1) # full attention
            subsequent_mask = torch.tril(full_mask)            # subsequent sequence should be invisible to each token position
            subsequent_mask = subsequent_mask.unsqueeze(0)     # add a batch dim (1, max_target_sequence_length, max_target_sequence_length)

            # The source mask is just the source pad mask.
            # The target mask is the intersection of the target pad mask and the subsequent_mask.
            return target_pad_mask & subsequent_mask

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        tok = {}

        # source into encoder
        input_en_prime = encoded_inputs['en_prime']
        src_tok = self.tokenizer.tokenize_src(input_en_prime, padding='max_length', truncation=True, max_length=self.input_max_length, return_tensors='pt')
        tok['encoder_input_ids'] = src_tok['input_ids']
        tok['encoder_input_mask'] = self.__create_masks(tok['encoder_input_ids'], pad_token_id=1, is_enc=True)

        input_keyword = encoded_inputs['keyword']
        keyword_tok = self.tokenizer.tokenize_src(input_keyword, padding='max_length', truncation=True, max_length=self.keyword_max_length, return_tensors='pt')
        tok['encoder_keyword_ids'] = keyword_tok['input_ids']
        tok['encoder_keyword_mask'] = self.__create_masks(tok['encoder_keyword_ids'], pad_token_id=1, is_enc=True)
        
        input_en = encoded_inputs['en']
        output_tok = self.tokenizer.tokenize_tgt(input_en, padding='max_length', truncation=True, max_length=self.output_max_length, return_tensors='pt')
        tok['decoder_tgt_ids'] = output_tok['input_ids']
        tok['decoder_tgt_mask'] = self.__create_masks(tok['decoder_tgt_ids'], pad_token_id=1, is_enc=False)
        tok['decoder_tgt_labels'] = self.__shift_tokens_left(tok['decoder_tgt_ids'], pad_token_id=1)

        input_fr = encoded_inputs['fr']
        translate_tok = self.tokenizer.tokenize_translate(input_fr, padding='max_length', truncation=True, max_length=self.translate_max_length, return_tensors='pt')
        tok['decoder_tsl_ids'] = translate_tok['input_ids']
        tok['decoder_tsl_mask'] = self.__create_masks(tok['decoder_tsl_ids'], pad_token_id=1, is_enc=False)
        tok['decoder_tsl_labels'] = self.__shift_tokens_left(tok['decoder_tsl_ids'], pad_token_id=1)
        
        return tok

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "valid", "test"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=24)
            )
        return res
