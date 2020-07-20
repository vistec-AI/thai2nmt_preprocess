# thai2nmt_preprocess


This repository includes scripts to clean texts and filter segment pairs used for constructing our English-Thai machine translation dataset.

We apply rule-based text cleaning to all texts obtained. Then, we filter out segments that are incorrectly aligned using handcrafted rules and [Google's Universal Sentence Encoder Multilingual](https://tfhub.dev/google/universal-sentence-encoder-multilingual/1) (USE). For the USE, we encode English-Thai segment pairs into 2 embbeddings, and compute cosine similarity. We then set the lower-bound of cosine similarity that will be included in our resulting dataset.

<br/>

__Required libraries:__

- [TensorFlow 2.x](https://github.com/tensorflow) (__tensorflow__)
- [TensorFlow Text](https://github.com/tensorflow/text) (__tensorflow_text__)
- [TensorFlow Hub](https://github.com/tensorflow/hub) (__tensorflow_hub__)
- [dask](https://github.com/dask/dask)


The following libraries can be install via `pip` as follows.

```bash
pip install -r requirements.txt
```

<br/>

## Text cleaning and segment pair filtering script

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

<br/>

### Example: Run `clean_subdataset.py` on mockup dataset.

1. Display the content of  mockup dataset.

     ```
     cat ./mockup.csv
     ```

     ```
     en_text,th_text
     Ok sounds good,โอเค ฟังดูเยี่ยม
     ไม่ห้าโมงก็สองทุ่มเลยค่ะ,5 or 8.
     "Hi, I'm looking to book a table for Korean fod.",สวัสดีค่ะ ช่วยจองร้านอาหารเกาหลีให้หน่อยได้มั้ยคะ?
     "Hi, I'm looking to book a table for Korean fod.",สวัสดีค่ะ ช่วยจองร้านอาหารเกาหลีให้หน่อยได้มั้ยคะ?
     Strengthening cooperation in the legal and judicial field,ส่งเสริมความร่วมมือด้านกฎหมายและด้านยุติธรรม
     "ASEAN and Japan engage with each other through various mechanisms at many levels. This includes the ASEAN-Japan Summit, the ASEAN-Japan Ministerial Meeting and the ASEAN-Japan Forum for Senior Officials. In 2011, Japan became the first Dialogue Partner to establish a Permanent Mission to ASEAN in Jakarta.",อาเซียนและญี่ปุ่นมีกลไกความร่วมมือหลายระดับ
     "ASEAN and Japan engage with each other through various mechanisms at many levels. This includes the ASEAN-Japan Summit, the ASEAN-Japan Ministerial Meeting and the ASEAN-Japan Forum for Senior Officials. In 2011, Japan became the first Dialogue Partner to establish a Permanent Mission to ASEAN in Jakarta.",อาเซียนและญี่ปุ่นมีกลไกความร่วมมือหลายระดับ อาทิ การประชุมสุดยอดอาเซียน-ญี่ปุ่น การประชุมรัฐมนตรีต่างประเทศอาเซียน-ญี่ปุ่น การประชุมอาเซียน-ญี่ปุ่นในระดับเจ้าหน้าที่อาวุโส ทั้งนี้ เมื่อปี 2554 ญี่ปุ่นเป็นประเทศคู่เจรจาของอาเซียนประเทศแรกที่จัดตั้งคณะผู้แทนถาวรประจำอาเซียน ณ กรุงจาการ์ตา
     "Yes, I concur.",ใช่ ข้าเห็นด้วยกับท่าน
     "I was thinking about the “Melting Pot“ since they have a special Anniversary dinner special.",กำลังคิดอยู่ว่าจะจองร้าน “เมล์ติ้ง พ็อท“ ค่ะ เพราะเขามีเมนูพิเศษสำหรับดินเนอร์วันครบรอบน่ะ
     "alomond mile and whipped cream",นมอัลมอนเเละ       วิปครีม
     "alomond mile &amp; whipped cream",นมอัลมอน &amp; วิปครีม
     ```

2. Run the text cleaning and segment filtering script .

     ```
     python clean_subdataset.py ./examples/mockup.csv \
     --drop_dup \
     --drop_na \
     --fix_html \
     --th_norm \
     --unicode_norm nfkc \
     --rm_useless_spaces \
     --drop_th_in_en \
     --drop_by_en_tok_count \
     --en_tok_min 2 \
     --en_tok_max 400 \
     --drop_by_th2en_ratio \
     --th2en_ratio_min 0.25 \
     --th2en_ratio_max 4 \
     --out_dir ./examples
     ```

     ```
     Loading CSV file from ./examples/mockup.csv
     filename: mockup (from: ./examples/mockup.csv)
     Current number of sentence pairs:       11


     [Text cleaning] Perform Thai text normalization.

     [Text cleaning] Perform fixing HTML special characters.

     [Text cleaning] Perform removing redundant spaces.

     [Filtering] Perform dropping rows contains NA or empty values:

     Remaining number of sentence pairs:       11 (remove      0 rows)

     [Filtering] Perform dropping rows that are duplicated:

     en duplicates:      2 segments
     th duplicates:      1 segments
     en,th duplicates:      1 segments

     Remaining number of sentence pairs:       10 (remove      1 rows)

     Calculating En tokens, percentage of En and Th character, Th characters in En segments

          th charcters in en texts: 1 segments
          en char (mean, median, min, max): 0.73, 0.82 (0.00-0.88)
          th char (mean, median, min, max): 0.88, 0.95 (0.29-1.00)
          en tokens (mean, median, min, max): 16.60, 8.0 (3-50)
          th tokens (mean, median, min, max): 12.20, 6.5 (3-49)
          th tokens_space (mean, median, min, max): 14.40, 7.5 (5-59)
          th-to-en tokens ratio (mean, median, min, max): 1.65, 1.01 (0.75-6.25)
          en sentences (mean, median, min, max): 1.40, 1.0 (1-3)
          th sentences (mean, median, min, max): 3.20, 2.0 (1-11)
     

     [Filtering] Perform dropping rows Thai characters appeared in English sentence:

     Remaining number of sentence pairs:        9 (remove     1 rows)

     [Filtering] Perform dropping rows that number of En tokens are out of range [2, 400]:

     Remaining number of sentence pairs:        9 (remove      0 rows)

     [Filtering] Perform dropping rows by Thai tokens and English tokens ratio is out of range [0.250000, 4.00] :

     Remaining number of sentence pairs:        8 (remove     1 rows)



     -----------------------------------

     Total number of sentence pairs:        8



     Begin writing result file to `./examples/mockup.cleaned.csv`
     Done.
     ```


3. Display the cleaned and filtered mockup dataset.

     ```
     cat ./examples/cleaned_dataset/mockup.cleaned.csv
     ```

     ```
     ,en_text,th_text
     0,Ok sounds good,โอเค ฟังดูเยี่ยม
     1,"Hi, I'm looking to book a table for Korean fod.",สวัสดีค่ะ ช่วยจองร้านอาหารเกาหลีให้หน่อยได้มั้ยคะ?
     2,Strengthening cooperation in the legal and judicial field,ส่งเสริมความร่วมมือด้านกฎหมายและด้านยุติธรรม
     3,"ASEAN and Japan engage with each other through various mechanisms at many levels. This includes the ASEAN-Japan Summit, the ASEAN-Japan Ministerial Meeting and the ASEAN-Japan Forum for Senior Officials. In 2011, Japan became the first Dialogue Partner to establish a Permanent Mission to ASEAN in Jakarta.",อาเซียนและญี่ปุ่นมีกลไกความร่วมมือหลายระดับ อาทิ การประชุมสุดยอดอาเซียน-ญี่ปุ่น การประชุมรัฐมนตรีต่างประเทศอาเซียน-ญี่ปุ่น การประชุมอาเซียน-ญี่ปุ่นในระดับเจ้าหน้าที่อาวุโส ทั้งนี้ เมื่อปี 2554 ญี่ปุ่นเป็นประเทศคู่เจรจาของอาเซียนประเทศแรกที่จัดตั้งคณะผู้แทนถาวรประจําอาเซียน ณ กรุงจาการ์ตา
     4,"Yes, I concur.",ใช่ ข้าเห็นด้วยกับท่าน
     5,I was thinking about the “Melting Pot“ since they have a special Anniversary dinner special.,กําลังคิดอยู่ว่าจะจองร้าน “เมล์ติ้ง พ็อท“ ค่ะ เพราะเขามีเมนูพิเศษสําหรับดินเนอร์วันครบรอบน่ะ
     6,alomond mile and whipped cream,นมอัลมอนและ วิปครีม
     7,alomond mile & whipped cream,นมอัลมอน & วิปครีม
     ```