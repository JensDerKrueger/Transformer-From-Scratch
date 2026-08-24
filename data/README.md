# How to get `leipzig1M.txt`

The file `leipzig1M.txt` is intentionally not stored in this repository. It has
about one million sentences and is too large for normal source control.

The official source for Leipzig corpora is the Leipzig Corpora Collection:

- project/documentation: https://wortschatz.uni-leipzig.de/en/documentation
- downloads: https://wortschatz.uni-leipzig.de/en/download
- direct archive server: https://downloads.wortschatz-leipzig.de/corpora/

The local file used while developing this project contains one plain sentence per
line. Leipzig archives usually contain a tab-separated sentence file with an id
column and a sentence column. The command below downloads an official English
Wikipedia corpus with one million sentences and keeps only the sentence column:

```sh
cd data
curl -fL https://downloads.wortschatz-leipzig.de/corpora/eng_wikipedia_2016_1M.tar.gz \
    | tar -xzf - -O 'eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-sentences.txt' \
    | cut -f2 > leipzig1M.txt
```

Other Leipzig corpora can be used as well. For example, replace the corpus id in
the URL and in the `tar` path with another official Leipzig id such as
`deu_wikipedia_2021_1M`.

Note: the already existing local `leipzig1M.txt` has no metadata left in it, so
we cannot reliably reconstruct the exact original archive from the file alone.
For the lessons, this is fine: the code expects only one text line per sentence.
