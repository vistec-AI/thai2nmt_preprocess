# thai2nmt_preprocess


This repository includes scripts to clean texts and filter segment pairs used for constructing our English-Thai machine translation dataset.

We apply rule-based text cleaning to all texts obtained. Then, we filter out segments that are incorrectly aligned using handcrafted rules and [Google's Universal Sentence Encoder Multilingual](https://tfhub.dev/google/universal-sentence-encoder-multilingual/1) (USE). For the USE, we encode English-Thai segment pairs into 2 embbeddings, and compute cosine similarity. We then set the lower-bound of cosine similarity that will be included in our resulting dataset.


__Required libraries:__

- [TensorFlow 2.x](https://github.com/tensorflow) (__tensorflow__)
- [TensorFlow Text](https://github.com/tensorflow/text) (__tensorflow_text__)
- [TensorFlow Hub](https://github.com/tensorflow/hub) (__tensorflow_hub__)
- [dask](https://github.com/dask/dask)


The following libraries can be install via `pip` as follows.

```bash
pip install -r requirements.txt
```


### Text cleaning and segment pair filtering script

We use the following script to perform text cleaning and specify the threaholds used for segment pairs filtering rules.

```
python clean_subdataset.py [input_path]
```

Positional Argument:

```
input_path  : Path to the En-Th segment pairs sub-dataset. File extension can be either .tsv, .csv, or .json)
```


Optional arugments:

```
--drop_dup              : Drop duplicated segment pairs")
--drop_na               : Drop rows with NA
--fix_html              : Format HTML special characters
--th_norm               : Perform Thai text normalization according to pythainlp.utils.normailize
--rm_useless_spaces     : Remove redundant spces
--drop_th_in_en         : Drop rows where Thai chacters found in English segment
--drop_by_en_tok_count  : Drop rows based on #tokens of English segment
--drop_by_th_tok_count  : Drop rows based on #tokens of Thaisegment

--drop_by_en_char_per   : Drop rows based on percentage of English characters
--drop_by_th_char_per   : Drop rows based on percentage of Thai characters
--drop_by_use_sim       : Use Universal Sentence Encoder (USE) Multiligual model to 
                          filter pairs of English-Thai segments.

--drop_by_th2en_ratio   : Drop rows based on ratio of Thai to English tokens.
--th2en_ratio_min       : Lower bound of the Thai to English tokens ratio. (default: 0.0)
--th2en_ratio_max       : Upper bound of the Thai to English tokens ratio. (default: 0.0)

--en_char_per           : Lower bound of the English character percentage in segment. (default: 0.0)
--th_char_per           : Upper bound of the Thai character percentage in segment. (default: 0.0)
--en_tok_min            : Lower bound of the English tokens in segment. (default: 0)
--en_tok_max            : Upper bound of the English tokens in segment. (default: 500)
--th_tok_min            : Lower bound of the Thai tokens in segment. (default: 0)
--th_tok_max            : Upper bound of the Thai tokens in segment. (default: 500)

--out_dir               : Directory to store the cleaned and filtered sub-dataset
                          (default: "./cleaned_dataset")

--unicode_norm          : Unicode normalization including "none", "nfkc", "nfd", 
                          "nfc" and "nfkd". (default: "none" )

--use_sim_threshold'    : The threashold of segment pairs similarity to accept 
                          (this can be specified interactively

--batch_size            : Batch size for USE Multilingual model inference. (default: 2048)

```