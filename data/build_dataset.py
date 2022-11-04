from datasets import load_dataset
import pandas as pd
from eda import eda
import re

def remove_symbol(sent: str) -> str:
    """
    Clean some symbols such as &*#@,;
    """
    x = re.sub("&|;|#|@|,|\*", "", sent)
    x = re.sub("\s+", " ", x)
    return x

if __name__ == "__main__":
    dataset = load_dataset('kde4', lang1='en', lang2='fr')
    en_prime = []
    fr = []
    keyword = []
    en = []
    for el in dataset['train']['translation']:
        en_sent = remove_symbol(el['en'])
        length = len([_ for _ in en_sent.split(' ')])
        if length < 3:
            continue
        augment_sent, key_word = eda(en_sent, alpha_rs=0.5, p_rd=0.3, p_kw=0.15)
        length = len([_ for _ in key_word.split(' ')])
        if length < 3:
            continue
        fr.append(remove_symbol(el['fr']))
        en.append(en_sent)
        en_prime.append(augment_sent)
        keyword.append(key_word)
    df = pd.DataFrame({'en': en, 'fr': fr, 'en_prime': en_prime, 'keyword': keyword})
    df.to_csv('data.csv', index=False)