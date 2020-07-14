import argparse
import re
from pathlib import Path
from functools import partial

import pandas as pd
import pythainlp
from pythainlp.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import sent_tokenize as en_sent_tokenize

word_tokenize_no_space = partial(word_tokenize, keep_whitespace=False)
word_tokenize_space = partial(word_tokenize, keep_whitespace=True)


def char_percent(pattern, text):
    return len(re.findall(pattern, text)) / (len(text)+0.01)


if __name__ == "__main__":

    print(f'PyThaiNLP version : {pythainlp.__version__}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    args = parser.parse_args()

    file_ext = Path(args.input_path).suffix
    
    if file_ext == '.csv':
        df = pd.read_csv(args.input_path)
    elif file_ext == '.txt':
        df = pd.read_csv(args.input_path, sep=',',
                         names=['th_text', 'en_text'])
    elif file_ext == '.tsv':
        df = pd.read_csv(args.input_path, sep='\t')
    else:
        raise("Invalid file extension (should be either .csv or .tsv)")

    # missing
    df['missing_en'] = df.en_text.isna()
    df['missing_th'] = df.th_text.isna()

    # characters
    df['per_en'] = df.en_text.map(
        lambda x: char_percent(r'[a-zA-Z0-9]', str(x)))
    df['per_th'] = df.th_text.map(lambda x: char_percent(r'[ก-๙0-9]', str(x)))
    df['th_in_en'] = df.en_text.map(
        lambda x: 1 if char_percent(r'[ก-๙]', str(x)) else 0)

    # tokens
    df['en_tokens'] = df.en_text.map(lambda x: len(str(x).split()))
    df['th_tokens'] = df.th_text.map(
        lambda x: len(word_tokenize_no_space(str(x))))
    df['th_tokens_space'] = df.th_text.map(
        lambda x: len(word_tokenize_space(str(x))))

    df['e2t_tokens'] = df.en_tokens / df.th_tokens

    # sentences
    df['en_sentences'] = df.en_text.map(
        lambda x: len(en_sent_tokenize(str(x))))
    df['th_sentences'] = df.th_text.map(lambda x: len(sent_tokenize(str(x))))

    print(f'''
    {args.input_path}
    shape: {df.shape}
    missing en: {df.missing_en.sum()} segments
    missing th: {df.missing_th.sum()} segments
    en duplicates: {df.en_text.count() - df.en_text.nunique()} segments
    th duplicates: {df.th_text.count() - df.th_text.nunique()} segments
    th charcters in en texts: {df.th_in_en.sum()} segments
    en char (mean, median, min, max): {df.per_en.mean():.2f}, {df.per_en.median():.2f} ({df.per_en.min():.2f}-{df.per_en.max():.2f})
    th char (mean, median, min, max): {df.per_th.mean():.2f}, {df.per_th.median():.2f} ({df.per_th.min():.2f}-{df.per_th.max():.2f})
    en tokens (mean, median, min, max): {df.en_tokens.mean():.2f}, {df.en_tokens.median()} ({df.en_tokens.min()}-{df.en_tokens.max()})
    th tokens [excluded space](mean, median, min, max): {df.th_tokens.mean():.2f}, {df.th_tokens.median()} ({df.th_tokens.min()}-{df.th_tokens.max()})
    th tokens [included space] (mean, median, min, max): {df.th_tokens_space.mean():.2f}, {df.th_tokens_space.median()} ({df.th_tokens_space.min()}-{df.th_tokens_space.max()})
    en-to-th tokens ratio (mean, median, min, max): {df.e2t_tokens.mean():.2f}, {df.e2t_tokens.median():.2f} ({df.e2t_tokens.min():.2f}-{df.e2t_tokens.max():.2f})
    en sentences (mean, median, min, max): {df.en_sentences.mean():.2f}, {df.en_sentences.median()} ({df.en_sentences.min()}-{df.en_sentences.max()})
    th sentences (mean, median, min, max): {df.th_sentences.mean():.2f}, {df.th_sentences.median()} ({df.th_sentences.min()}-{df.th_sentences.max()})
    ''')
