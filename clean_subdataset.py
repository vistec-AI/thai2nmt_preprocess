import argparse
import json
import html
import re
import os
import multiprocessing
import unicodedata

from pathlib import Path
from functools import partial
from typing import Pattern

import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.ulmfit import rm_useless_spaces, fix_html
from pythainlp.util import normalize as th_normalize
from pythainlp.tokenize import word_tokenize
from pythainlp.tokenize import sent_tokenize as th_sent_tokenize
from nltk.tokenize import sent_tokenize as en_sent_tokenize


dask.config.set(scheduler='processes')

word_tokenize_no_space = partial(word_tokenize, keep_whitespace=False)
word_tokenize_space = partial(word_tokenize, keep_whitespace=True)

USE_MODEL_NAME = os.getenv(
    'USE_MODEL') or 'universal-sentence-encoder-multilingual/3'


def char_percent(pattern: Pattern, text: str):
    return len(re.findall(pattern, text)) / (len(text)+0.01)


def get_similar_score(lang1: str, lang2: str, batch_size: int, embed):

    scores = []

    if len(lang1) % batch_size != 0:
        num_of_batch = int(len(lang1)/batch_size)+1
    else:
        num_of_batch = int(len(lang1)/batch_size)

    for i in range(num_of_batch):
        start = i*batch_size
        end = start+batch_size
        if i <= num_of_batch:

            lang1_temp = lang1[start:end]
            lang2_temp = lang2[start:end]

            lang1_embedding = embed(lang1_temp)
            lang2_embedding = embed(lang2_temp)
            distance_matrix = tf.matmul(
                lang1_embedding, lang2_embedding, transpose_b=True).numpy()

            for j in range(len(distance_matrix)):
                scores.append(distance_matrix[j][j])

    return scores


def main(args):
    if args.input_path == '':
        raise "Positional argument: `input_path` is not specified"

    input_path = args.input_path
    input_file_name = Path(args.input_path).stem
    input_file_ext = Path(args.input_path).suffix

    if input_file_ext == '.csv':
        print(f'Loading CSV file from {input_path}')
        df = pd.read_csv(input_path, index_col=[])
    elif input_file_ext == '.tsv':
        print(f'Loading TSV file from {input_path}')
        df = pd.read_csv(input_path, sep='\t', index_col=[])

    elif input_file_ext == '.json':
        print(f'Loading JSON file from {input_path}')

        objs_list = json.load(open(input_path))
        objs_list = sum(objs_list, [])
        # break new lines
        dataset = []
        for item in tqdm(objs_list, total=len(objs_list)):
            en_lst = item['en_text'].split('\n')
            th_lst = item['th_text'].split('\n')
            # expand segment paris by splitting with `\n` if length between two side are equal
            if len(en_lst) == len(th_lst):
                for i, j in zip(en_lst, th_lst):
                    if len(i) != 0 and len(j) != 0:
                        dataset.append({'en_text': i, 'th_text': j})
            else:
                # if not equal, add those segments as a segment pair
                if len(item['en_text']) != 0 and len(item['th_text']) != 0:

                    dataset.append(
                        {'en_text': item['en_text'], 'th_text': item['th_text']})

        df = pd.DataFrame(dataset)
    else:
        raise "Invalid file extension"

    print(f'filename: {input_file_name} (from: {input_path})')
    print(f'Current number of sentence pairs: {df.shape[0]:8,}\n')

    df['en_text'] = df['en_text'].apply(str)
    df['th_text'] = df['th_text'].apply(str)

    df['en_text'] = df['en_text'].apply(lambda x: x.replace(u'\u200b', u'').replace(
        u'\x99', u'').replace(u'\x9c', u'').replace(u'\xa0', u' '))
    df['th_text'] = df['th_text'].apply(lambda x: x.replace(u'\u200b', u'').replace(
        u'\x99', u'').replace(u'\x9c', u'').replace(u'\xa0', u' '))

    df_dd = dd.from_pandas(df, npartitions=2)

    if args.unicode_norm != 'none':
        norm_type = args.unicode_norm.upper()
        print(
            f'\n[Text cleaning] Perform Unicode Text Normalization (type: {norm_type}).')
        df_dd['en_text'] = df_dd['en_text'].apply(lambda x: unicodedata.normalize(
            norm_type, str(x)), meta=('en_text', 'str')).compute()
        df_dd['th_text'] = df_dd['th_text'].apply(lambda x: unicodedata.normalize(
            norm_type, str(x)), meta=('th_text', 'str')).compute()

    if args.th_norm:

        print('\n[Text cleaning] Perform Thai text normalization.')
        df_dd['th_text'] = df_dd['th_text'].apply(
            lambda x: th_normalize(x), meta=('th_text', 'str')).compute()

    if args.fix_html:

        print('\n[Text cleaning] Perform fixing HTML special characters.')
        df_dd['en_text'] = df_dd['en_text'].apply(lambda x: html.unescape(
            x).replace('& # 34;', '"'), meta=('en_text', 'str')).compute()
        df_dd['th_text'] = df_dd['th_text'].apply(lambda x: html.unescape(
            x).replace('& # 34;', '"'), meta=('th_text', 'str')).compute()

    if args.rm_useless_spaces:
        # from pythainlp.ulmfit.rm_useless_spaces
        print('\n[Text cleaning] Perform removing redundant spaces.')
        df_dd['en_text'] = df_dd['en_text'].apply(lambda x: rm_useless_spaces(str(x).replace(
            '\n', ' ').replace('\t', ' ').replace('\r', '').strip()), meta=('en_text', 'str')).compute()
        df_dd['th_text'] = df_dd['th_text'].apply(lambda x: rm_useless_spaces(str(x).replace(
            '\n', ' ').replace('\t', ' ').replace('\r', '').strip()), meta=('th_text', 'str')).compute()

    if args.drop_na:

        print('\n[Filtering] Perform dropping rows contains NA or empty values:\n')
        temp_len = len(df_dd.index)
        df_dd = df_dd.dropna().reset_index(drop=True)

        df_dd = df_dd[df_dd['en_text'].str.strip().astype(bool)]
        df_dd = df_dd[df_dd['th_text'].str.strip().astype(bool)]

        print(
            f' Remaining number of sentence pairs: {len(df_dd.index):8,} (remove {temp_len - len(df_dd.index):6,} rows)')

    if args.drop_dup:
        temp_len = len(df_dd.index)

        print('\n[Filtering] Perform dropping rows that are duplicated:\n')

        print(
            f' en duplicates: {df.en_text.count() - df.en_text.nunique():6,} segments')
        print(
            f' th duplicates: {df.th_text.count() - df.th_text.nunique():6,} segments')

        en_th_nuniq = df.groupby(['en_text', 'th_text']).ngroups
        print(f' en,th duplicates: {df.shape[0] - en_th_nuniq:6,} segments')

        df_dd = df_dd.drop_duplicates(
            subset=['th_text', 'en_text'], keep='first').reset_index(drop=True)

        print(
            f'\n Remaining number of sentence pairs: {len(df_dd.index):8,} (remove {temp_len - len(df_dd.index):6,} rows)')

    df = df_dd.compute()

    print('\nCalculating En tokens, percentage of En and Th character, Th characters in En segments')
    df['en_tokens'] = df_dd.en_text.apply(lambda x: len(
        word_tokenize_no_space(str(x))), meta=('en_text', 'str')).compute()
    df['th_tokens'] = df_dd.th_text.apply(lambda x: len(
        word_tokenize_no_space(str(x))), meta=('th_text', 'str')).compute()
    df['th_tokens_space'] = df_dd.th_text.apply(lambda x: len(
        word_tokenize_space(str(x))), meta=('th_text', 'str')).compute()

    df['per_en'] = df_dd.en_text.apply(lambda x: char_percent(
        r'[a-zA-Z0-9]', x), meta=('en_text', 'str')).compute()
    df['per_th'] = df_dd.th_text.apply(lambda x: char_percent(
        r'[ก-๙0-9]', x), meta=('th_text', 'str')).compute()
    df['th_in_en'] = df_dd.en_text.apply(lambda x: 1 if char_percent(
        r'[ก-๙]', x) else 0, meta=('th_text', 'str')).compute()
    df['t2e_tokens'] = df.en_tokens / df.th_tokens

    # sentences
    df['en_sentences'] = df_dd.en_text.apply(lambda x: len(
        en_sent_tokenize(str(x))), meta=('en_text', 'str')).compute()
    df['th_sentences'] = df_dd.th_text.apply(lambda x: len(
        th_sent_tokenize(str(x))), meta=('th_text', 'str')).compute()

    print(f'''
     th charcters in en texts: {df.th_in_en.sum()} segments
     en char (mean, median, min, max): {df.per_en.mean():.2f}, {df.per_en.median():.2f} ({df.per_en.min():.2f}-{df.per_en.max():.2f})
     th char (mean, median, min, max): {df.per_th.mean():.2f}, {df.per_th.median():.2f} ({df.per_th.min():.2f}-{df.per_th.max():.2f})
     en tokens (mean, median, min, max): {df.en_tokens.mean():.2f}, {df.en_tokens.median()} ({df.en_tokens.min()}-{df.en_tokens.max()})
     th tokens (mean, median, min, max): {df.th_tokens.mean():.2f}, {df.th_tokens.median()} ({df.th_tokens.min()}-{df.th_tokens.max()})
     th tokens_space (mean, median, min, max): {df.th_tokens_space.mean():.2f}, {df.th_tokens_space.median()} ({df.th_tokens_space.min()}-{df.th_tokens_space.max()})
     th-to-en tokens ratio (mean, median, min, max): {df.t2e_tokens.mean():.2f}, {df.t2e_tokens.median():.2f} ({df.t2e_tokens.min():.2f}-{df.t2e_tokens.max():.2f})
     en sentences (mean, median, min, max): {df.en_sentences.mean():.2f}, {df.en_sentences.median()} ({df.en_sentences.min()}-{df.en_sentences.max()})
     th sentences (mean, median, min, max): {df.th_sentences.mean():.2f}, {df.th_sentences.median()} ({df.th_sentences.min()}-{df.th_sentences.max()})
    ''')

    if args.drop_th_in_en:
        print(
            '\n[Filtering] Perform dropping rows Thai characters appeared in English sentence:\n')
        temp_len = df.shape[0]
        df = df[df.th_in_en == 0].reset_index(drop=True)

        print(
            f' Remaining number of sentence pairs: {df.shape[0]:8,} (remove {temp_len - df.shape[0]:5,} rows)')

    if args.drop_by_th_tok_count:
        print(
            f'\n[Filtering] Perform dropping rows that number of Th tokens (included space token) are out of range [{args.th_tok_min}, {args.th_tok_max}]:\n')
        temp_len = df.shape[0]
        df = df[(df.th_tokens_space >= args.th_tok_min) & (
            df.th_tokens_space <= args.th_tok_max)].reset_index(drop=True)
        print(
            f' Remaining number of sentence pairs: {df.shape[0]:8,} (remove {temp_len - df.shape[0]:6,} rows)')

    if args.drop_by_en_tok_count:
        print(
            f'\n[Filtering] Perform dropping rows that number of En tokens are out of range [{args.en_tok_min}, {args.en_tok_max}]:\n')
        temp_len = df.shape[0]
        df = df[(df.en_tokens >= args.en_tok_min) & (
            df.en_tokens <= args.en_tok_max)].reset_index(drop=True)
        print(
            f' Remaining number of sentence pairs: {df.shape[0]:8,} (remove {temp_len - df.shape[0]:6,} rows)')

    if args.drop_by_en_char_per:
        print(
            f'\n[Filtering] Perform dropping rows that character percentage of En characters less than {args.en_char_per:2f}:\n')
        temp_len = df.shape[0]
        df = df[(df.per_en >= args.en_char_per)].reset_index(drop=True)
        print(
            f' Remaining number of sentence pairs: {df.shape[0]:8,} (remove {temp_len - df.shape[0]:5,} rows)')

    if args.drop_by_th_char_per:
        print(
            f'\n[Filtering] Perform dropping rows that character perfoence of Th characters less than {args.th_char_per:2f}:\n')
        temp_len = df.shape[0]
        df = df[(df.per_th >= args.th_char_per)
                ].reset_index().reset_index(drop=True)
        print(
            f' Remaining number of sentence pairs: {df.shape[0]:8,} (remove {temp_len - df.shape[0]:5,} rows)')

    if args.drop_by_th2en_ratio:
        print(
            f'\n[Filtering] Perform dropping rows by Thai tokens and English tokens ratio is out of range [{args.th2en_ratio_min:2f}, {args.th2en_ratio_max:.2f}] :\n')

        temp_len = df.shape[0]

        df = df[(df.t2e_tokens >= args.th2en_ratio_min) & (
            df.t2e_tokens <= args.th2en_ratio_max)].reset_index().reset_index(drop=True)

        print(
            f' Remaining number of sentence pairs: {df.shape[0]:8,} (remove {temp_len - df.shape[0]:5,} rows)')

    if args.drop_by_use_sim:
        use_temp_filepath = f'./temp/{input_file_name}.use.csv'
        if os.path.exists(use_temp_filepath) == True:
            print(
                f'\n Found calculated USE similairy results in `{use_temp_filepath}`')

            print(f'\n Begin loading the cached file instead.')

            df = pd.read_csv(use_temp_filepath, index_col=[0])

        else:
            print(
                f'\nCalculating sentence pairs similariry based on {USE_MODEL_NAME}')
            # check parallel texts with universal sentence encoder

            print('\n Loading USE model to memory...')
            try:
                _emb_model = hub.load(
                    f'https://tfhub.dev/google/{USE_MODEL_NAME}')
            except Exception as e:
                print('[ERROR] Can\'t load USE model.')
                raise e
            print('_emb_model:', _emb_model)
            print('\n Done.')

            bs = args.batch_size
            use_sim = []
            print('\n Start vectorizeing sentence pairs and calculate cosine similarity.')
            for i in tqdm(range(df.shape[0]//bs+1)):
                df_dd = df.iloc[i * bs:(i + 1) * bs]

                en = df_dd.en_text
                th = df_dd.th_text

                scores = get_similar_score(en, th, bs, _emb_model)

                use_sim += scores

            df['use_sim'] = use_sim

            del df_dd
            del _emb_model
            del use_sim

            print('\n Done.\n')
            print(f'\n USE similarity distribution:')

            print(df['use_sim'].value_counts(bins=10))

            print('\nStore dataframe with USE similarirty results.')

            df.to_csv(f'./temp/{input_file_name}.use.csv')

        print('\n Number of sentences will by filtered given a threshold:\n')
        for idx, i in enumerate(np.arange(0, 1.0, 0.1)):

            print(
                f'  - Threshold {i:.2f}: {(df.shape[0] - len(df[df.use_sim > i])):5,} rows (remains {len(df[df.use_sim > i]):6,})')

            print(
                f'\tSample 10 sentence pairs with threahold between ({i}, {i + 0.1}):')
            sampled_df = df[((df.use_sim > i) & (df.use_sim < i + 0.1))]
            for k, v in sampled_df.sample(min(10, sampled_df.shape[0]), random_state=1234).iterrows():
                print(
                    f'\t en: {v["en_text"]}\n\t th: {repr(v["th_text"])}\n score: {v["use_sim"]:.4f}\n')
            print('')
            print('-'*25)
            print('\n')

        # Python input as float
        if args.use_sim_threshold is None:
            print(
                '\n\n [Manual mode] Please enter the threshold to opt out sentence pairs: ')

            threshold = float(input())
            print(f'\n The threshold manually selected is {threshold:.2f}')

        else:

            threshold = args.use_sim_threshold
            assert type(threshold) == float
            print(
                f'\n The threshold selected based on `args` is {threshold:.2f}')

        temp_len = df.shape[0]
        df = df[df.use_sim > threshold]

        print(
            f'\n Remaining number of sentence pairs: {df.shape[0]:8,} (remove {temp_len - df.shape[0]:5,} rows)')

    print('\n\n')
    print('-'*35)
    print(f'\n Total number of sentence pairs: {df.shape[0]:8,}')

    print('\n')
    if args.drop_by_use_sim:
        output_path = os.path.join(
            args.out_dir, f'{input_file_name}.cleaned.use_threshold-{threshold}.csv')
    else:
        output_path = os.path.join(
            args.out_dir, f'{input_file_name}.cleaned.csv')
    print(f'\nBegin writing result file to `{output_path}`')

    df[['en_text', 'th_text']].to_csv(output_path, encoding='utf-8')

    print('Done.')
    print('')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_path', help="Path to the En-Th segment pairs sub-dataset.")
    parser.add_argument('--drop_dup', action='store_true',
                        help="Drop duplicated segment pairs")
    parser.add_argument('--drop_na', action='store_true',
                        help="Drop rows with NA")
    parser.add_argument('--fix_html', action='store_true',
                        help="Format HTML special characters")
    parser.add_argument('--th_norm', action='store_true',
                        help="Perform Thai text normalization according to pythainlp.utils.normailize")
    parser.add_argument('--rm_useless_spaces',
                        action='store_true', help='Remove redundant spces')
    parser.add_argument('--drop_th_in_en', action='store_true',
                        help='Drop rows where Thai chacters found in English segment')
    parser.add_argument('--drop_by_en_tok_count', action='store_true',
                        help='Drop rows based on #tokens of English segment')
    parser.add_argument('--drop_by_th_tok_count', action='store_true',
                        help='Drop rows based on #tokens of Thaisegment')

    parser.add_argument('--drop_by_en_char_per', action='store_true',
                        help='Drop rows based on percentage of English characters')
    parser.add_argument('--drop_by_th_char_per', action='store_true',
                        help='Drop rows based on percentage of Thai characters')
    parser.add_argument('--drop_by_use_sim', action='store_true',
                        help='Use Universal Sentence Encoder (USE) Multiligual model to filter pairs of English-Thai segment')

    parser.add_argument('--drop_by_th2en_ratio', action='store_true',
                        help='Drop rows based on ratio of Thai to English tokens.')
    parser.add_argument('--th2en_ratio_min', type=float, default=0.0,
                        help='Lower bound of the Thai to English tokens ratio.')
    parser.add_argument('--th2en_ratio_max', type=float, default=15.0,
                        help='Upper bound of the Thai to English tokens ratio.')

    parser.add_argument('--en_char_per', type=float, default=0.0,
                        help='Lower bound of the English character percentage in segment.')
    parser.add_argument('--th_char_per', type=float, default=0.0,
                        help='Upper bound of the Thai character percentage in segment.')
    parser.add_argument('--en_tok_min', type=int, default=0,
                        help='Lower bound of the English tokens in segment.')
    parser.add_argument('--en_tok_max', type=int, default=500,
                        help='Upper bound of the English tokens in segment.')
    parser.add_argument('--th_tok_min', type=int, default=0,
                        help='Lower bound of the Thai tokens in segment.')
    parser.add_argument('--th_tok_max', type=int, default=500,
                        help='Upper bound of the Thai tokens in segment.')

    parser.add_argument('--out_dir', type=str, default='./cleaned_dataset',
                        help='Drectory to store the cleaned and filtered sub-dataset')

    parser.add_argument('--unicode_norm', type=str, default='none',
                        help='Unicode normalization including [none, nfkc, nfd, nfc, nfkd]')

    parser.add_argument('--use_sim_threshold', type=float, default=None,
                        help='The threashold of segment pairs similarity to accept (Can be specified interactively)')

    parser.add_argument("--batch_size", default=2048, type=int,
                        help='Batch size for USE Multilingail model inference.')

    args = parser.parse_args()

    main(args)
