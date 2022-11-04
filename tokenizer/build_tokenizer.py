import argparse, sys
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from collections.abc import Mapping
from datasets import load_dataset
from tokenizers import decoders
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing

def build_tokenizer(lang: str, vocab_size: int, is_target: bool, dir: str):
    dataset = load_dataset('csv', data_files='../data/data.csv')

    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))

    if is_target:
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", 2),
                ("<eos>", 3)
            ],
        )

    def batch_iterator(dataset, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            examples = dataset[i: i + batch_size]
            if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
                encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
            else:
                encoded_inputs = examples
            yield encoded_inputs[lang]


    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
    )

    tokenizer.train_from_iterator(batch_iterator(dataset['train']), trainer=trainer, length=len(dataset['train']))

    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=256)
    tokenizer.decoder = decoders.WordPiece()

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )

    if is_target:
        wrapped_tokenizer.save_pretrained(f"{dir}/{lang}_tokenizer_tgt")
    else:
        wrapped_tokenizer.save_pretrained(f"{dir}/{lang}_tokenizer_src")

def str2bool(input: str) -> bool:
    if input == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--lang", help="Language to build tokenizer for")
    parser.add_argument("--vocab_size", help="Vocab size")
    parser.add_argument("--is_target", help="Is the language input works as a target lang")
    parser.add_argument("--dir", help="Directory to store")
    args=parser.parse_args()
    build_tokenizer(
        lang=args.lang,
        vocab_size=int(args.vocab_size),
        is_target=str2bool(args.is_target),
        dir=args.dir)