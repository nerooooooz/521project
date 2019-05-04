# 521project

## Dataset

`msr_paraphrase_data.txt` is the whole data set.
`msr_paraphrase_train.txt` is the train.
`msr_paraphrase_test.txt` is the test.

## Code

`SimilarityVectorizer.py` is the class for vector-based method.
`WordnetSimilarity.py` is the class for wordnet-based method.
`main.py` is the main class containing string similarity and the logistic regression for all.

### Running Sequence:
* Just run main.py is ok, it might request you to download the wordnet_ic package.
* Use `import nltk \\ nltk.download("wordnet_ic")` to download.

* Or you can run `SimilarityVectorizer.py` and `WordnetSimilarity.py` two modules first and then run `main.py`.

## 50K_GoogleNews_vecs.txt
You also need to have this file for word2vec vectorizer.

Download [here](https://drive.google.com/file/d/1VKz_8FFTQebHIL-Ok_Qo63rwhR6dbu4G/view?usp=sharing).

A truncated version of the 300-vectors trained by Google Research on an enormous corpus of Google News.
Only the first 50K are circulated here to reduce memory and disk usage; 
the full file is available at <https://code.google.com/archive/p/word2vec/> .
