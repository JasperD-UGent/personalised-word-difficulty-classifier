# personalised-word-difficulty-classifier

This repository contains the code to train the personalised word difficulty classifier for learners of Spanish as a foreign/second language (L2), presented in [anonymised_reference]. The repository allows you to train two different versions of the classifier: a *base* version (trained on the original version of the LexComSpaL2 corpus; Degraeuwe & Goethals, 2024) and a *word family* version (trained on an enriched version of LexComSpaL2 in which information on word family knowledge was added; for more details see the [requirements section](#requirements) below). Information on how to run the scripts is also provided below, in two separate subsections.

## Requirements

First, install the Python packages included in `requirements.txt`, preferably in a fresh virtual environment. The scripts were tested with Python 3.11 on a Linux machine. To install the requirements via the command line, run the following command in your shell (from the root directory of this repository as your working directory):

```shell
$ python -m pip install -r requirements.txt
```

**NOTE**: The `pydot` package (used to create the visualisation of the classifier's architecture) is a Python interface for [Graphviz](https://graphviz.org/). In case the script throws an error related to the `pydot` package, you might need to install Graphviz separately. Especially on Windows machines this error might occur.

**NOTE**: If you want to train the classifier on a GPU, please consult [TensorFlow's dedicated web page](https://www.tensorflow.org/install/pip) to install the correct version for your device and CUDA architecture.

Secondly, [download](https://zenodo.org/records/3255001) the pretrained fastText embeddings used by the classifier (see also corresponding [GitHub repository](https://github.com/BotCenter/spanishWordEmbeddings) for more details). For the classifier presented in [anonymous_reference], the L model was used.

Thirdly, the original LexComSpaL2 data as well as the "word-family enriched" version of the dataset need to be downloaded. The original data can be retrieved by cloning the corresponding GitHub repository from the command line as follows (note that the data need to be saved into the `input_v1/LexComSpaL2` folder):

```shell
$ git clone https://github.com/JasperD-UGent/LexComSpaL2.git input_v1/LexComSpaL2
```

The enriched version of LexComSpaL2 needs to be downloaded manually from the corresponding [anonymous GitHub repository](https://anonymous.4open.science/r/LexComSpaL2-enriched-word-families-101F) and stored in `input_v1/LexComSpaL2_enriched`. Upon acceptance of the BEA paper (which has hitherto been referred to as "[anonymous_reference]"), this part of the read me will be replaced by a `git clone` command.


## Base classifier

To train the base classifier, run the following command in your shell (replacing "[PATH_FASTTEXT_EMBEDDINGS]" by the path pointing to the file that contains the fastText embeddings on your local device):

```shell
$ python train_classifier_base_v1.py [PATH_FASTTEXT_EMBEDDINGS]
```

For more information on the optional parameters that can be specified (i.e. the device on which the script should be run, the number of cross-validation folds to run, the number of epochs to run, and the batch size), have a look at the source code or run the command below.

```shell
$ python train_classifier_base_v1.py -h
```

For example, to run the script on CPU, for 5 cross-validation folds, for 25 epochs, and a batch size of 32, this would be the command:

```shell
$ python train_classifier_base_v1.py [PATH_FASTTEXT_EMBEDDINGS] --device cpu --n_folds_cv 5 --n_epochs 25 --batch_size 32
```

**NOTE**: Instead of the full argument names, it is also possible to use the short versions (`-d`, `-f`, `-e`, and `-b` respectively).

## Word family classifier

To train the word family classifier, run the command below in your shell. In the default setting, only the word family level (i.e. "token", "lemma", "source", or "combi") and the path to the file containing the pretrained fastText static word embeddings need to be specified. In the command below, the classifier is trained on "token" as the word family level.

```shell
$ python train_classifier_wordFams_v1.py token [PATH_FASTTEXT_EMBEDDINGS]
```

For more information on the optional parameters that can be specified (i.e. the device on which the script should be run, the number of cross-validation folds to run, the number of epochs to run, and the batch size), have a look at the source code or run the command below.

```shell
$ python train_classifier_wordFams_v1.py -h
```

## References
- [anonymous_reference]

- Degraeuwe, J., & Goethals, P. (2024). LexComSpaL2: A Lexical Complexity Corpus for Spanish as a Foreign Language. In N. Calzolari, M.-Y. Kan, V. Hoste, A. Lenci, S. Sakti, & N. Xue (Eds.), *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)* (pp. 10432–10447). Torino, Italy: ELRA and ICCL. https://aclanthology.org/2024.lrec-main.912/
