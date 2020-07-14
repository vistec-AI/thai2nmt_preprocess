# thai2nmt_preprocess


This repository includes scripts to clean texts and filter segment pairs used for constructing our English-Thai machine translation dataset.

We apply rule-based text cleaning to all texts obtained. Then, we filter out segments that are incorrectly aligned using handcrafted rules and [Google's Universal Sentence Encoder Multilingual](https://tfhub.dev/google/universal-sentence-encoder-multilingual/1) (USE). For the USE, we encode English-Thai segment pairs into 2 embbeddings, and compute cosine similarity. We then set the lower-bound of cosine similarity that will be included in our resulting dataset.


__Required libraries:__

- TensorFlow v2.x (__tensorflow__)
- TensorFlow Text (__tensorflow_text__)
