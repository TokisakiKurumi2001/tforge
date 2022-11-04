from transformers import AutoTokenizer

class TForgeTokenizer:
    def __init__(self, source_tokenizer_ck: str, target_tokenizer_ck: str, translate_tokenizer_ck: str):
        self.source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_ck)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_ck)
        self.translate_tokenizer = AutoTokenizer.from_pretrained(translate_tokenizer_ck)

    def tokenize_src(self, sentences, **kwargs):
        return self.source_tokenizer(sentences, **kwargs)
    
    def tokenize_tgt(self, sentences, **kwargs):
        return self.target_tokenizer(sentences, **kwargs)

    def tokenize_translate(self, sentences, **kwargs):
        return self.translate_tokenizer(sentences, **kwargs)