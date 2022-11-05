from TForge import TForgeForConditionalGeneration, TForgeModel
from transformers import AutoTokenizer
model = TForgeModel.from_pretrained('TForge_model/v1')
gen = TForgeForConditionalGeneration(model)
source_tok = AutoTokenizer.from_pretrained('tokenizer/v1/en_tokenizer_src')
sent = "I went to the market to buy some food"
keyword = "went market food"
input_ids = source_tok([sent], padding="max_length", truncation=True, max_length=40, return_tensors='pt').input_ids
keyword_ids = source_tok([keyword], padding="max_length", truncation=True, max_length=15, return_tensors='pt').input_ids
output = gen.generate(input_ids, keyword_ids)[0]
tokenizer = AutoTokenizer.from_pretrained('tokenizer/v1/en_tokenizer_tgt')
reconstruct = tokenizer.decode(output, skip_special_tokens=False)
print(reconstruct)