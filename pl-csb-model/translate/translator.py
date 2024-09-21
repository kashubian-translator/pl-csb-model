import re

from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

def normalize_translation(input: list[str]) -> str:
    # Technically matching against 'Latn' will break if we do languages other than PL and CSB but we won't
    return re.match(r'^.*Latn (.*)<.*$', input[0]).group(1)


def translate(text: str, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, src_lang='pol_Latn', tgt_lang='csb_Latn', a=32, b=3, max_input_length=1024, num_beams=4, **kwargs) -> list[str]:
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    model.eval()
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result)