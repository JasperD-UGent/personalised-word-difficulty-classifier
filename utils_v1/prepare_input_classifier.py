import sys
from typing import Dict, Optional, Tuple


def define_features_train_val_test(
        d_dataset_annots: Dict, d_dataset_split: Dict, fold: str, n_partps: int,
        d_char_embs: Dict, d_fasttext_embs: Dict, d_partp_id_embs: Dict, d_prof_level_embs: Dict,
        d_n_years_exp_embs: Dict, d_l1_embs: Dict, d_word_fam_embs: Optional[Dict], d_labels: Dict
) -> Tuple[Dict, Dict, Dict]:
    """Link features to the set (training, validation, or test) they belong to in a given fold.
    :param d_dataset_annots: Dictionary containing the LexComSpaL2 annotations.
    :param d_dataset_split: Dictionary containing the predefined LexComSpaL2 dataset splits.
    :param fold: The cross-validation fold for which the classifier is being trained, in string format.
    :param n_partps: The number of participants.
    :param d_char_embs: Dictionary containing the features for the "character embedding" subtype.
    :param d_fasttext_embs: Dictionary containing the features for the "fastText embedding" subtype.
    :param d_partp_id_embs: Dictionary containing the features for the "participant ID" subtype.
    :param d_prof_level_embs: Dictionary containing the features for the "proficiency level" subtype.
    :param d_n_years_exp_embs: Dictionary containing the features for the "years of experience" subtype.
    :param d_l1_embs: Dictionary containing the features for the "L1" subtype.
    :param d_word_fam_embs: Dictionary containing the features for the "word family" subtype.
    :param d_labels: Dictionary containing the true labels to train and evaluate the classifier.
    :return: A dictionary containing the concatenated features (per subtype) for each of the three sets.
    """
    x_train_inp_char = []
    x_train_inp_fasttext = []
    x_train_inp_partp_id = []
    x_train_inp_prof_level = []
    x_train_inp_n_years_exp = []
    x_train_inp_l1 = []
    x_train_inp_word_fam = []
    y_train = []
    y_train_int = []

    x_val_inp_char = []
    x_val_inp_fasttext = []
    x_val_inp_partp_id = []
    x_val_inp_prof_level = []
    x_val_inp_n_years_exp = []
    x_val_inp_l1 = []
    x_val_inp_word_fam = []
    y_val = []

    x_test_inp_char = []
    x_test_inp_fasttext = []
    x_test_inp_partp_id = []
    x_test_inp_prof_level = []
    x_test_inp_n_years_exp = []
    x_test_inp_l1 = []
    x_test_inp_word_fam = []
    y_test = []
    l_sent_ids_test = []

    for sent in d_dataset_annots:

        if d_dataset_split[sent]["split"][fold] == "train":
            x_train_inp_char += [d_char_embs[sent][partp] for partp in d_char_embs[sent]]
            x_train_inp_fasttext += [d_fasttext_embs[sent][partp] for partp in d_fasttext_embs[sent]]
            x_train_inp_partp_id += [d_partp_id_embs[sent][partp] for partp in d_partp_id_embs[sent]]
            x_train_inp_prof_level += [d_prof_level_embs[sent][partp] for partp in d_prof_level_embs[sent]]
            x_train_inp_n_years_exp += [d_n_years_exp_embs[sent][partp] for partp in d_n_years_exp_embs[sent]]
            x_train_inp_l1 += [d_l1_embs[sent][partp] for partp in d_l1_embs[sent]]
            x_train_inp_word_fam += [
                d_word_fam_embs[sent][partp] for partp in d_word_fam_embs[sent]
            ] if d_word_fam_embs is not None else []
            y_train += [d_labels[sent][partp] for partp in d_labels[sent]]
            y_train_int += [
                (d_dataset_annots[sent]["annots_per_token"][idx][partp] - 1)
                for idx in d_dataset_annots[sent]["annots_per_token"]
                for partp in d_dataset_annots[sent]["annots_per_token"][idx]
            ]

        if d_dataset_split[sent]["split"][fold] == "val":
            x_val_inp_char += [d_char_embs[sent][partp] for partp in d_char_embs[sent]]
            x_val_inp_fasttext += [d_fasttext_embs[sent][partp] for partp in d_fasttext_embs[sent]]
            x_val_inp_partp_id += [d_partp_id_embs[sent][partp] for partp in d_partp_id_embs[sent]]
            x_val_inp_prof_level += [d_prof_level_embs[sent][partp] for partp in d_prof_level_embs[sent]]
            x_val_inp_n_years_exp += [d_n_years_exp_embs[sent][partp] for partp in d_n_years_exp_embs[sent]]
            x_val_inp_l1 += [d_l1_embs[sent][partp] for partp in d_l1_embs[sent]]
            x_val_inp_word_fam += [
                d_word_fam_embs[sent][partp] for partp in d_word_fam_embs[sent]
            ] if d_word_fam_embs is not None else []
            y_val += [d_labels[sent][partp] for partp in d_labels[sent]]

        if d_dataset_split[sent]["split"][fold] == "test":
            x_test_inp_char += [d_char_embs[sent][partp] for partp in d_char_embs[sent]]
            x_test_inp_fasttext += [d_fasttext_embs[sent][partp] for partp in d_fasttext_embs[sent]]
            x_test_inp_partp_id += [d_partp_id_embs[sent][partp] for partp in d_partp_id_embs[sent]]
            x_test_inp_prof_level += [d_prof_level_embs[sent][partp] for partp in d_prof_level_embs[sent]]
            x_test_inp_n_years_exp += [d_n_years_exp_embs[sent][partp] for partp in d_n_years_exp_embs[sent]]
            x_test_inp_l1 += [d_l1_embs[sent][partp] for partp in d_l1_embs[sent]]
            x_test_inp_word_fam += [
                d_word_fam_embs[sent][partp] for partp in d_word_fam_embs[sent]
            ] if d_word_fam_embs is not None else []
            y_test += [d_labels[sent][partp] for partp in d_labels[sent]]
            l_sent_ids_test += [sent for _ in range(n_partps)]
            
    d_train = {
        "x_char": x_train_inp_char,
        "x_fasttext": x_train_inp_fasttext,
        "x_partp_id": x_train_inp_partp_id,
        "x_prof_level": x_train_inp_prof_level,
        "x_n_years_exp": x_train_inp_n_years_exp,
        "x_l1": x_train_inp_l1,
        "x_word_fam": x_train_inp_word_fam,
        "y": y_train,
        "y_int": y_train_int
    }
    d_val = {
        "x_char": x_val_inp_char,
        "x_fasttext": x_val_inp_fasttext,
        "x_partp_id": x_val_inp_partp_id,
        "x_prof_level": x_val_inp_prof_level,
        "x_n_years_exp": x_val_inp_n_years_exp,
        "x_l1": x_val_inp_l1,
        "x_word_fam": x_val_inp_word_fam,
        "y": y_val,
    }
    d_test = {
        "x_char": x_test_inp_char,
        "x_fasttext": x_test_inp_fasttext,
        "x_partp_id": x_test_inp_partp_id,
        "x_prof_level": x_test_inp_prof_level,
        "x_n_years_exp": x_test_inp_n_years_exp,
        "x_l1": x_test_inp_l1,
        "x_word_fam": x_test_inp_word_fam,
        "y": y_test,
        "l_sent_ids": l_sent_ids_test
    }

    return d_train, d_val, d_test
