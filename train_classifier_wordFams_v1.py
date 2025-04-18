from utils_v1.prepare_input_classifier import define_features_train_val_test
from utils_v1.process_JSONs import load_json
from utils_v1.process_LexComSpaL2_enriched import add_lexcomspal2_data_to_dictionaries
from utils_v1.select_CPU_GPU import select_device
from train_classifier_base_v1 import load_lexcomspal2
import argparse
import csv
import fasttext
import keras
import numpy as np
import os
import random
from sklearn.metrics import f1_score, matthews_corrcoef, mean_squared_error, root_mean_squared_error
from sklearn.utils import class_weight
import statistics
import sys
import tensorflow as tf
import torch
from tqdm import tqdm
from typing import Dict, Tuple


# set fixed random seed to ensure reproducible results
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
torch.random.manual_seed(SEED)


def load_lexcomspal2_enriched_annots(word_fam_level: str) -> Dict:
    """Load annotation data from enriched LexComSpaL2 corpus.
    :param word_fam_level: Word family level.
    :return: A dictionary containing the LexComSpaL2 annotations (sentence>target word>participant).
    """
    path_direc_lexcomspal2 = os.path.join("input_v1", "LexComSpaL2_enriched")
    d_dataset_annots_combi = {}

    if word_fam_level in ["token", "combi"]:
        fn = "LexComSpaL2_all_enriched_wordFamily_token.tsv"
        d_dataset_annots_token = {}
        add_lexcomspal2_data_to_dictionaries(
            path_direc_lexcomspal2, fn, "token", d_dataset_annots_token, d_dataset_annots_combi
        )

        if word_fam_level == "token":
            return d_dataset_annots_token
        
    if word_fam_level in ["lemma", "combi"]:
        fn = "LexComSpaL2_all_enriched_wordFamily_lemma.tsv"
        d_dataset_annots_lemma = {}
        add_lexcomspal2_data_to_dictionaries(
            path_direc_lexcomspal2, fn, "lemma", d_dataset_annots_lemma, d_dataset_annots_combi
        )

        if word_fam_level == "lemma":
            return d_dataset_annots_lemma

    if word_fam_level in ["source", "combi"]:
        fn = "LexComSpaL2_all_enriched_wordFamily_source.tsv"
        d_dataset_annots_source = {}
        add_lexcomspal2_data_to_dictionaries(
            path_direc_lexcomspal2, fn, "source", d_dataset_annots_source, d_dataset_annots_combi
        )

        if word_fam_level == "source":
            return d_dataset_annots_source

    if word_fam_level == "combi":
        return d_dataset_annots_combi


def define_features_to_train_classifier(
        device: str, d_chars_to_idxs: Dict, d_dataset_annots: Dict, d_partp_feats: Dict, path_ft_vecs: str,
        n_partps: int, n_l1s: int, max_sent_length: int, max_word_length: int, len_word_fam_emb: int
) -> Tuple[Dict, Dict]:
    """Define features (i.e. vectors) to train personalised word difficulty classifier.
    :param device: Device on which the classifier should be trained.
    :param d_chars_to_idxs: Dictionary in which characters are mapped to indices (used to train the convolutional
        character embedding model).
    :param d_dataset_annots: Dictionary containing the LexComSpaL2 annotations.
    :param d_partp_feats: Dictionary containing the participant features.
    :param path_ft_vecs: Path to file containing the pretrained fastText vectors.
    :param n_partps: Number of participants.
    :param n_l1s: Number of different L1s among the participants.
    :param max_sent_length: Maximum sentence length among sentences in LexComSpaL2 corpus.
    :param max_word_length: Maximum word length among target words in LexComSpaL2 corpus.
    :param len_word_fam_emb: Length of the word family embedding.
    :return: A dictionary containing the features and a dictionary containing the corresponding true labels (i.e.
        lexical complexity prediction annotations).
    """
    # load pretrained fastText vectors
    ft_vecs = fasttext.load_model(path_ft_vecs)

    # define features
    with tf.device(device):
        d_char_embeddings = {}
        d_fasttext_embeddings = {}
        d_partp_id_embeddings = {}
        d_prof_level_embeddings = {}
        d_n_years_experience_embeddings = {}
        d_l1_embeddings = {}
        d_word_fam_embeddings = {}
        d_labels = {}

        for partp in tqdm([loop + 1 for loop in range(n_partps)], desc="Define features per participant ..."):

            for sent in d_dataset_annots:
                l_toks = d_dataset_annots[sent]["sent_text"].split()

                # empty features for every word in sentence, to be overwritten later on
                l_inputs_char = [tf.zeros([max_word_length], tf.float32) for _ in range(max_sent_length)]
                l_inputs_fasttext = [tf.zeros([300], tf.float32) for _ in range(max_sent_length)]
                l_inputs_subj = [tf.zeros([n_partps], tf.float32) for _ in range(max_sent_length)]
                l_inputs_prof_lev = [tf.zeros([3], tf.float32) for _ in range(max_sent_length)]
                l_inputs_n_years_exp = [tf.zeros([1], tf.float32) for _ in range(max_sent_length)]
                l_inputs_l1 = [tf.zeros([n_l1s], tf.float32) for _ in range(max_sent_length)]
                l_inputs_word_fam = [tf.zeros([len_word_fam_emb], tf.float32) for _ in range(max_sent_length)]

                # empty labels for every word in sentence, to be overwritten later on (non-target words are assigned a
                # label of 5, target words are assigned the value of their original lexical complexity prediction value
                # minus one)
                l_labels_sent = [np.float32(5) for _ in range(max_sent_length)]

                for idx, tok in enumerate(l_toks):

                    if idx in d_dataset_annots[sent]["annots_per_token"]:

                        # character embedding
                        l_inputs_tok_np = np.zeros((max_word_length,), dtype=np.float32)

                        for idx_char, char in enumerate(tok):
                            l_inputs_tok_np[idx_char] = np.float32(d_chars_to_idxs[char])

                        l_inputs_char[idx] = tf.convert_to_tensor(l_inputs_tok_np)

                        # word embedding
                        ft_vec = torch.from_numpy(ft_vecs.get_word_vector(tok))
                        ft_vec_np = ft_vec.numpy()
                        ft_vec_tf = tf.convert_to_tensor(ft_vec_np)
                        partp_id_vec_np = np.zeros((n_partps,), dtype=np.float32)
                        partp_id_vec_np[(partp - 1)] = 1.
                        partp_id_vec_tf = tf.convert_to_tensor(partp_id_vec_np, tf.float32)
                        prof_lev_vec_np = np.zeros((3,), dtype=np.float32)
                        prof_lev_vec_np[d_partp_feats[partp]["proficiency"]] = 1.
                        prof_lev_vec_tf = tf.convert_to_tensor(prof_lev_vec_np, tf.float32)
                        n_years_exp_vec = tf.convert_to_tensor(
                            np.array([d_partp_feats[partp]["years_experience"]]), tf.float32
                        )

                        if n_l1s == 1:
                            l1_vec_tf = tf.convert_to_tensor(
                                np.array([d_partp_feats[partp]["native_language"]]), tf.float32
                            )
                        else:
                            l1_vec_np = np.zeros((n_l1s,), dtype=np.float32)
                            l1_vec_np[(d_partp_feats[partp]["native_language"])] = 1.
                            l1_vec_tf = tf.convert_to_tensor(prof_lev_vec_np, tf.float32)

                        word_fam_vec_np = np.zeros((len_word_fam_emb,), dtype=np.float32)
                        idx_word_fam_emb = 0

                        for word_fam_level in d_dataset_annots[sent]["annots_per_word_fam_level"]:
                            d_annots_word_fam = d_dataset_annots[sent]["annots_per_word_fam_level"][word_fam_level]

                            for entry in [
                                "multiple_occurrences", "stat_sign",
                                "annots_multiple_occ_min", "annots_multiple_occ_max"
                            ]:
                                
                                if entry == "multiple_occurrences":

                                    if d_annots_word_fam[entry][idx]:
                                        word_fam_vec_np[idx_word_fam_emb] = 1.
                                    else:
                                        word_fam_vec_np[(idx_word_fam_emb + 1)] = 1.

                                    idx_word_fam_emb += 2

                                elif entry == "stat_sign":

                                    if d_annots_word_fam[entry][idx] is None:
                                        word_fam_vec_np[(idx_word_fam_emb + 2)] = 1.
                                    else:

                                        if d_annots_word_fam[entry][idx]:
                                            word_fam_vec_np[idx_word_fam_emb] = 1.
                                        else:
                                            word_fam_vec_np[(idx_word_fam_emb + 1)] = 1.

                                    idx_word_fam_emb += 3

                                else:

                                    if d_annots_word_fam[entry][idx] is None:
                                        word_fam_vec_np[(idx_word_fam_emb + 5)] = 1.
                                    else:
                                        annot_min_or_max = d_annots_word_fam[entry][idx][partp]
                                        word_fam_vec_np[(idx_word_fam_emb + annot_min_or_max - 1)] = 1.

                                    idx_word_fam_emb += 6

                        word_fam_vec_tf = tf.convert_to_tensor(word_fam_vec_np, tf.float32)

                        #   - overwrite empty features for target word
                        l_inputs_fasttext[idx] = ft_vec_tf
                        l_inputs_subj[idx] = partp_id_vec_tf
                        l_inputs_prof_lev[idx] = prof_lev_vec_tf
                        l_inputs_n_years_exp[idx] = n_years_exp_vec
                        l_inputs_l1[idx] = l1_vec_tf
                        l_inputs_word_fam[idx] = word_fam_vec_tf

                        #   - overwrite empty label for target word
                        label = d_dataset_annots[sent]["annots_per_token"][idx][partp]
                        l_labels_sent[idx] = np.float32((int(label) - 1))

                for dic in [
                    d_char_embeddings, d_fasttext_embeddings, d_partp_id_embeddings, d_prof_level_embeddings,
                    d_n_years_experience_embeddings, d_l1_embeddings, d_word_fam_embeddings, d_labels
                ]:

                    if sent not in dic:
                        dic[sent] = {}

                # features
                d_char_embeddings[sent][partp] = tf.convert_to_tensor(l_inputs_char)
                d_fasttext_embeddings[sent][partp] = tf.convert_to_tensor(l_inputs_fasttext)
                d_partp_id_embeddings[sent][partp] = tf.convert_to_tensor(l_inputs_subj)
                d_prof_level_embeddings[sent][partp] = tf.convert_to_tensor(l_inputs_prof_lev)
                d_n_years_experience_embeddings[sent][partp] = tf.convert_to_tensor(l_inputs_n_years_exp)
                d_l1_embeddings[sent][partp] = tf.convert_to_tensor(l_inputs_l1)
                d_word_fam_embeddings[sent][partp] = tf.convert_to_tensor(l_inputs_word_fam)

                # labels (= lexical complexity difficulty prediction value assigned by participants)
                d_labels[sent][partp] = tf.convert_to_tensor(l_labels_sent)

        d_feats = {
            "char_embeddings": d_char_embeddings,
            "fastText_embeddings": d_fasttext_embeddings,
            "partp_ID_embeddings": d_partp_id_embeddings,
            "prof_level_embeddings": d_prof_level_embeddings,
            "n_years_experience_embeddings": d_n_years_experience_embeddings,
            "L1_embeddings": d_l1_embeddings,
            "word_fam_embeddings": d_word_fam_embeddings
        }

        return d_feats, d_labels


def train_classifier(
        word_fam_level: str, device: str, d_chars_to_idxs: Dict, d_dataset_annots: Dict, d_dataset_split: Dict,
        n_partps: int, n_l1s: int, max_sent_length: int, max_word_length: int, len_word_fam_emb: int,
        d_feats: Dict, d_labels: Dict, n_folds_cv: int, n_epochs: int, batch_size: int
) -> None:
    """Train BiLSTM word difficulty classifier in K-fold cross-validation setup.
    :param word_fam_level: Word family level.
    :param device: Device on which the classifier should be trained.
    :param d_chars_to_idxs: Dictionary in which characters are mapped to indices (used to train the convolutional
        character embedding model).
    :param d_dataset_annots: Dictionary containing the LexComSpaL2 annotations.
    :param d_dataset_split: Dictionary containing the predefined LexComSpaL2 dataset splits.
    :param n_partps: Number of participants.
    :param n_l1s: Number of different L1s among the participants.
    :param len_word_fam_emb: Length of the word family embedding.
    :param max_sent_length: Maximum sentence length among sentences in LexComSpaL2 corpus.
    :param max_word_length: Maximum word length among target words in LexComSpaL2 corpus.
    :param d_feats: Dictionary containing the features to train the classifier.
    :param d_labels: Dictionary containing the true labels to train and evaluate the classifier.
    :param n_folds_cv: Number of cross-validations folds to run.
    :param n_epochs: Number of epochs (i.e. iterations over the entire training set) to run within each cross-validation
        fold.
    :param batch_size: Size of the batches the training data is split into within every epoch.
    :return: `None`
    """
    # prepare directory structure to save classifier outputs
    current_setup = f"{n_folds_cv}-CV_{n_epochs}-E_{batch_size}-BS"

    direc_outp = "output_v1"

    direc_classifier = f"classifier_{current_setup}"
    path_direc_visualisation = os.path.join(direc_outp, direc_classifier, "visualisation")
    path_plot = os.path.join(path_direc_visualisation, f"classifier_wordFam_{word_fam_level}_architecturePlot.png")
    path_direc_weights = os.path.join(direc_outp, direc_classifier, "weights")

    direc_performance = f"performance_{current_setup}"
    path_direc_performance = os.path.join(direc_outp, direc_performance)

    for path in [path_direc_visualisation, path_direc_weights, path_direc_performance]:

        if not os.path.isdir(path):
            os.makedirs(path)

    # train classifier
    d_performance = {}

    with tf.device(device):

        # extract separate features from overarching dictionary
        d_char_embs = d_feats["char_embeddings"]
        d_fasttext_embs = d_feats["fastText_embeddings"]
        d_partp_id_embs = d_feats["partp_ID_embeddings"]
        d_prof_level_embs = d_feats["prof_level_embeddings"]
        d_n_years_exp_embs = d_feats["n_years_experience_embeddings"]
        d_l1_embs = d_feats["L1_embeddings"]
        d_word_fam_embs = d_feats["word_fam_embeddings"]

        # perform K-fold cross-validation
        for fold in range(n_folds_cv):
            print(f"\n\n----- Performing cross-validation fold {fold + 1} / {n_folds_cv} -----\n")
            d_feats_train, d_feats_val, d_feats_test = define_features_train_val_test(
                d_dataset_annots, d_dataset_split, str(fold), n_partps,
                d_char_embs, d_fasttext_embs, d_partp_id_embs, d_prof_level_embs, d_n_years_exp_embs, d_l1_embs,
                d_word_fam_embs, d_labels
            )

            # determine class weights
            class_weights_orig = class_weight.compute_class_weight(
                "balanced", classes=np.unique(d_feats_train["y_int"]), y=d_feats_train["y_int"]
            )
            d_class_weights = {idx: weight for idx, weight in enumerate(class_weights_orig)}
            d_class_weights[5] = 0.
            d_class_weights_print = {idx: float(round(d_class_weights[idx], 2)) for idx in d_class_weights}
            print(f"\t- Class weights: {d_class_weights_print}.\n")

            # define layers of the classifier

            #   - layers at character level (i.e. the convolutional character embedding model)
            inp_char_1 = tf.keras.layers.Input((max_sent_length, max_word_length), name="char_inp")
            emb = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Embedding(len(d_chars_to_idxs) + 1, 16), name="char_emb"
            )(inp_char_1)
            conv_1 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv1D(filters=4, kernel_size=3, strides=1, padding="same"), name="char_conv1"
            )(emb)
            max_1 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPool1D(pool_size=2), name="char_max1"
            )(conv_1)
            conv_2 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv1D(filters=4, kernel_size=3, strides=1, padding="same"), name="char_conv2"
            )(max_1)
            max_2 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPool1D(pool_size=2), name="char_max2"
            )(conv_2)
            flat = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten(), name="char_flat"
            )(max_2)
            output_char = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(6, activation="softmax"), name="aux_out"
            )(flat)

            #   - layers at word level (i.e. the BiLSTM architecture)
            inp_word_1 = keras.layers.Input((max_sent_length, 300), name="fasttext")
            inp_word_2 = keras.layers.Input((max_sent_length, n_partps), name="subj_inp")
            inp_word_3 = keras.layers.Input((max_sent_length, 3), name="prof_lev_inp")
            inp_word_4 = keras.layers.Input((max_sent_length, 1), name="n_years_exp_inp")
            inp_word_5 = keras.layers.Input((max_sent_length, n_l1s), name="l1")
            inp_word_6 = keras.layers.Input((max_sent_length, len_word_fam_emb), name="word_fam")
            concat_1 = keras.layers.Concatenate(name="concat_1")(
                [flat, inp_word_1, inp_word_2, inp_word_3, inp_word_4, inp_word_5, inp_word_6]
            )
            mask = keras.layers.Masking(mask_value=0., name="mask")(concat_1)
            bilstm = keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), name="bilstm")(mask)
            concat_2 = keras.layers.Concatenate(
                name="concat_2"
            )([bilstm, inp_word_2, inp_word_3, inp_word_4, inp_word_5, inp_word_6])
            output_word = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(6, activation="softmax"), name="main_out"
            )(concat_2)
            classifier = keras.models.Model(
                inputs=[inp_char_1, inp_word_1, inp_word_2, inp_word_3, inp_word_4, inp_word_5, inp_word_6],
                outputs=[output_word, output_char]
            )

            if fold == 0:
                print(classifier.summary())

            #   - save visual representation of classifier architecture
            if not os.path.exists(path_plot):
                tf.keras.utils.plot_model(classifier, to_file=path_plot, show_shapes=True, show_layer_names=True)

            #   - compile and fit classifier
            classifier.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(),
                optimizer="adam",
                metrics=[keras.metrics.SparseCategoricalAccuracy()]
            )
            history = classifier.fit(
                [tf.convert_to_tensor(d_feats_train["x_char"]), tf.convert_to_tensor(d_feats_train["x_fasttext"]),
                 tf.convert_to_tensor(d_feats_train["x_partp_id"]), tf.convert_to_tensor(d_feats_train["x_prof_level"]),
                 tf.convert_to_tensor(d_feats_train["x_n_years_exp"]), tf.convert_to_tensor(d_feats_train["x_l1"]),
                 tf.convert_to_tensor(d_feats_train["x_word_fam"])],
                tf.convert_to_tensor(d_feats_train["y"]),
                epochs=n_epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=(
                    [tf.convert_to_tensor(d_feats_val["x_char"]), tf.convert_to_tensor(d_feats_val["x_fasttext"]),
                     tf.convert_to_tensor(d_feats_val["x_partp_id"]), tf.convert_to_tensor(d_feats_val["x_prof_level"]),
                     tf.convert_to_tensor(d_feats_val["x_n_years_exp"]), tf.convert_to_tensor(d_feats_val["x_l1"]),
                     tf.convert_to_tensor(d_feats_val["x_word_fam"])],
                    tf.convert_to_tensor(d_feats_val["y"])
                ),
                class_weight=d_class_weights,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
            )

            classifier.save(os.path.join(path_direc_weights, f"classifier_wordFam_{word_fam_level}_fold{fold}.keras"))
            print(f"\t- Number of epochs run: {len(history.history['val_loss'])} / {n_epochs}.")
            print(f"\t- Validation loss: {history.history['val_loss']}.\n")

            # evaluate classifier on test data
            d_idx_sent_id_test = {idx: sent for idx, sent in enumerate(d_feats_test["l_sent_ids"])}
            yhat = classifier.predict(
                [tf.convert_to_tensor(d_feats_test["x_char"]), tf.convert_to_tensor(d_feats_test["x_fasttext"]),
                 tf.convert_to_tensor(d_feats_test["x_partp_id"]), tf.convert_to_tensor(d_feats_test["x_prof_level"]),
                 tf.convert_to_tensor(d_feats_test["x_n_years_exp"]), tf.convert_to_tensor(d_feats_test["x_l1"]),
                 tf.convert_to_tensor(d_feats_test["x_word_fam"])],
                verbose=0
            )
            
            n_correct = 0
            n_incorrect = 0
            l_true_labels = []
            l_predicted_labels = []
            l_words_predicted_as_non_target = []
            d_probs_d_apostrophe = {(loop + 1): {"true": [], "false": []} for loop in range(5)}
            d_predictions_per_sent = {}

            #   - loop over predictions, which are located in main output layer at position 0 in `yhat` variable
            for idx_sent, sent in enumerate(yhat[0]):
                sent_id = d_idx_sent_id_test[idx_sent]

                if sent_id not in d_predictions_per_sent:
                    d_predictions_per_sent[sent_id] = {}

                for idx_tok, tok in enumerate(sent):

                    # NOTE: recall that, since the starting value in Python is 0, the original lexical complexity 
                    # prediction labels of 1 to 5 were mapped to the 0-4 range and the value of 5 was used to label
                    # non-target words
                    if int(d_feats_test["y"][idx_sent][idx_tok].numpy()) != 5:

                        # fallback strategy: if predicted as non-target (i.e. 5), take label 1 as prediction
                        tok_text = d_dataset_annots[sent_id]["sent_text"].split()[idx_tok]
                        predicted_prov = np.argmax(tok, axis=-1) if np.argmax(tok, axis=-1) != 5 else None
                        predicted = (predicted_prov + 1) if predicted_prov is not None else 1

                        if predicted_prov is None:
                            l_words_predicted_as_non_target.append((sent_id, tok_text))
                        
                        # append expected and predicted label to corresponding lists, which will be used to
                        # calculate MCC, F1, and (R)MSE metrics
                        expected = int(d_feats_test["y"][idx_sent][idx_tok].numpy()) + 1
                        l_true_labels.append(expected)
                        l_predicted_labels.append(predicted)

                        # preparatory steps for calculation of D' metric
                        for label in d_probs_d_apostrophe:

                            if label == expected:
                                d_probs_d_apostrophe[label]["true"].append(tok[label - 1])
                            else:
                                d_probs_d_apostrophe[label]["false"].append(tok[label - 1])

                        # preparatory steps for calculation of accuracy metric
                        if expected == predicted:
                            n_correct += 1
                        else:
                            n_incorrect += 1

                        # store results per sentence
                        if idx_tok not in d_predictions_per_sent[sent_id]:
                            d_predictions_per_sent[sent_id][idx_tok] = []

                        d_predictions_per_sent[sent_id][idx_tok].append((expected, predicted))

            #   - store results per participant
            d_predictions_per_partp = {}

            for sent in d_predictions_per_sent:

                for idx_tok in d_predictions_per_sent[sent]:

                    for partp_id_prov, tup in enumerate(d_predictions_per_sent[sent][idx_tok]):
                        partp_id = partp_id_prov + 1
                        true_label = tup[0]
                        predicted_label = tup[1]

                        if partp_id not in d_predictions_per_partp:
                            d_predictions_per_partp[partp_id] = {"true_labels": [], "predicted_labels": []}

                        d_predictions_per_partp[partp_id]["true_labels"].append(true_label)
                        d_predictions_per_partp[partp_id]["predicted_labels"].append(predicted_label)

            # calculate metrics
            
            #   - D'
            l_abs_diffs_d_apostrophe = []

            for label in d_probs_d_apostrophe:
                l_abs_diffs_d_apostrophe.append(
                    abs(statistics.mean(d_probs_d_apostrophe[label]["true"]) - statistics.mean(
                        d_probs_d_apostrophe[label]["false"]))
                )

            d_apostrophe = sum(l_abs_diffs_d_apostrophe) / 5
            print(f"\n\t- D' = {d_apostrophe} (absolute differences: {l_abs_diffs_d_apostrophe})")

            #   - MCC (Matthews correlation coefficient)
            mcc = matthews_corrcoef(l_true_labels, l_predicted_labels)
            print(f"\t- Matthews correlation coefficient = {mcc}")

            #   - weighted F1
            weighted_f1 = f1_score(l_true_labels, l_predicted_labels, average='weighted', zero_division=0)
            print(f"\t- Weighted F1 = {weighted_f1}")
            
            #   - mean squared error (MSE) and root mean squared error (RSME)
            mse = mean_squared_error(l_true_labels, l_predicted_labels)
            rmse = root_mean_squared_error(l_true_labels, l_predicted_labels)
            print(f"\t- MSE = {mse}")
            print(f"\t- RSME = {rmse}")

            #   - accuracy
            acc = n_correct / (n_correct + n_incorrect)
            print(f"\t- Accuracy = {acc}")

            # store performance results in dictionary
            d_performance[fold] = {
                "D'": d_apostrophe, "MCC": mcc, "weighted_F1": weighted_f1, "MSE": mse, "RMSE": rmse, "accuracy": acc
            }

    # write performance results to TSV
    fn_tsv = f"performance_per_fold_wordFam_{word_fam_level}.tsv"

    with open(os.path.join(path_direc_performance, fn_tsv), "w", newline="") as f_tsv:
        writer = csv.writer(f_tsv, delimiter="\t", lineterminator="\n")
        writer.writerow(["fold", "D'", "MCC", "weighted_F1", "MSE", "RMSE", "accuracy"])

        for fold in d_performance:
            writer.writerow([
                str(fold),
                str(d_performance[fold]["D'"]),
                str(d_performance[fold]["MCC"]),
                str(d_performance[fold]["weighted_F1"]),
                str(d_performance[fold]["MSE"]),
                str(d_performance[fold]["RMSE"]),
                str(d_performance[fold]["accuracy"]),
                ])

    f_tsv.close()


def main(
        word_fam_level: str, path_fasttext_vectors: str, device: str, n_folds_cv: int, n_epochs: int, batch_size: int
) -> None:
    """Entry point for the script. Will load the original LexComSpaL2 corpus and train the BiLSTM classifier.
    :param word_fam_level: Word family level. Choose between: 'token', 'lemma', 'source', and 'combi'. More details can
        be found in the read me.
    :param path_fasttext_vectors: Path to file containing the pretrained fastText vectors.
    :param device: Device on which the classifier should be trained. Enter 'cpu' for CPU and 'gpu:[ID_GPU]' for GPU
        (e.g., 'gpu:0' or 'gpu:1'). Defaults to 'gpu:0'.
    :param n_folds_cv: Number of cross-validations folds to run. Note: the original LexComSpaL2 dataset provides ten
        different training-validation-test splits, allowing for a maximum of ten folds. Defaults to 10 (the maximum
        number of folds allowed).
    :param n_epochs: Number of epochs (i.e. iterations over the entire training set) to run within each cross-validation
        fold.
    :param batch_size: Size of the batches the training data is split into within every epoch.
    :return: `None`
    """
    if "cpu" in device:
        prefer_gpu = False
        id_gpu = None
    elif "gpu" in device:
        prefer_gpu = True
        id_gpu = int(device.split(":")[-1])
    else:
        raise ValueError(
            f"Invalid device: {device}. Please check the `--help` of the argument parser for more information on the "
            f"`--device` argument."
        )

    if n_folds_cv > 10:
        print(f"Setting number of cross-validation folds to the maximum number of 10 ({n_folds_cv} given).")
        n_folds_cv = 10

    # select device on which the classifier should be trained
    device = select_device(prefer_gpu, id_gpu)
    print(f"Selected device: {device}.")

    # load LexComSpaL2 dataset and dictionary in which characters from LexComSpaL2 data are mapped to indices (used to
    # train the convolutional character embedding model)
    d_dataset_annots = load_lexcomspal2_enriched_annots(word_fam_level)  # new
    _, d_partp_feats, d_dataset_split = load_lexcomspal2()  # same as original dataset
    d_chars_to_idxs = load_json(os.path.join("input_v1", "d_chars_to_idxs.json"))

    # build dictionary containing features (i.e. vectors) to train the classifier
    n_partps = len(d_partp_feats)
    n_l1s = len(set([d_partp_feats[partp]["native_language"] for partp in d_partp_feats]))
    max_sent_length = max([len(d_dataset_annots[sent]["sent_text"].split()) for sent in d_dataset_annots])
    max_word_length = max(
        [len(tok) for sent in d_dataset_annots for tok in d_dataset_annots[sent]["sent_text"].split()]
    )
    len_word_fam_emb = 17 if word_fam_level != "combi" else (3 * 17)
    d_feats, d_labels = define_features_to_train_classifier(
        device, d_chars_to_idxs, d_dataset_annots, d_partp_feats, path_fasttext_vectors,
        n_partps, n_l1s, max_sent_length, max_word_length, len_word_fam_emb
    )

    # train BiLSTM classifier
    train_classifier(
        word_fam_level, device, d_chars_to_idxs, d_dataset_annots, d_dataset_split,
        n_partps, n_l1s, max_sent_length, max_word_length, len_word_fam_emb,
        d_feats, d_labels, n_folds_cv, n_epochs, batch_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "word_fam_level",
        choices=["token", "lemma", "source", "combi"], type=str,
        help="Word family level for which the script needs to be run."
    )
    parser.add_argument(
        "path_fasttext_vectors",
        help="Path to file containing the pretrained fastText vectors."
    )
    parser.add_argument(
        "-d", "--device",
        default="gpu:0", type=str,
        help="Device on which the classifier should be trained. Enter 'cpu' for CPU and 'gpu:[ID_GPU]' for GPU (e.g., "
             "'gpu:0' or 'gpu:1')."
    )
    parser.add_argument(
        "-f", "--n_folds_cv",
        default=10, type=int,
        help="Number of cross-validations folds to run. Note: the original LexComSpaL2 dataset provides ten different "
             "training-validation-test splits, allowing for a maximum of ten folds."
    )
    parser.add_argument(
        "-e", "--n_epochs",
        default=10, type=int,
        help="Number of epochs (i.e. iterations over the entire training set) to run within each cross-validation fold."
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=64, type=int,
        help="Size of the batches the training data is split into within every epoch."
    )
    main(**vars(parser.parse_args()))
