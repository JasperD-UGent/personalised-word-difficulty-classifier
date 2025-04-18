from ast import literal_eval
import csv
import os
from pathlib import Path
import sys
from typing import Dict, Union


def add_lexcomspal2_data_to_dictionaries(
        path_direc_lexcomspal2: Union[str, Path], fn: str, word_fam_level: str,
        d_dataset_annots: Dict, d_dataset_annots_combi: Dict
) -> None:
    """Add word family information included in enriched LexComSpaL2 to dictionaries.
    :param path_direc_lexcomspal2: Path to directory containing enriched LexComSpaL2 corpus.
    :param fn: Name of the file containing the enriched dataset version for a given word family level.
    :param word_fam_level: Word family level.
    :param d_dataset_annots: Dictionary in which the annotations for the given word family level should be stored.
    :param d_dataset_annots_combi: Dictionary in which the annotations for all word family levels combined should be
        stored.
    :return: `None`
    """
    with open(os.path.join(path_direc_lexcomspal2, fn), mode="r", encoding="utf-8") as f_annots:
        f_tsv_annots = csv.reader(f_annots, delimiter="\t")
        _ = next(f_tsv_annots)
        l_rows = [row for row in f_tsv_annots]

        for row in l_rows:
            id_line = row[0]
            idx_domain = id_line.split("_")[0]
            idx_sent = id_line.split("_")[1]
            idx_word = int(id_line.split("_")[2])
            d_annots_tok = literal_eval(row[5])
            multiple_occ = literal_eval(row[6])
            stat_sign = literal_eval(row[7]) if row[7] != "N/A" else None
            d_annots_min = literal_eval(row[8]) if row[8] != "N/A" else None
            d_annots_max = literal_eval(row[9]) if row[9] != "N/A" else None
            sent_id = f"{idx_domain}_{idx_sent}"

            for dic in [d_dataset_annots, d_dataset_annots_combi]:
            
                if sent_id not in dic:
                    dic[sent_id] = {
                        "sent_text": row[2], "annots_per_token": {},
                        "annots_per_word_fam_level": {}
                    }

                if word_fam_level not in dic[sent_id]["annots_per_word_fam_level"]:
                    dic[sent_id]["annots_per_word_fam_level"][word_fam_level] = {}

                for entry in [
                    "multiple_occurrences", "stat_sign",
                    "annots_multiple_occ_min", "annots_multiple_occ_max"
                ]:

                    if entry not in dic[sent_id]["annots_per_word_fam_level"][word_fam_level]:
                        dic[sent_id]["annots_per_word_fam_level"][word_fam_level][entry] = {}

                dic[sent_id]["annots_per_token"][idx_word] = d_annots_tok
                dic[sent_id]["annots_per_word_fam_level"][word_fam_level]["multiple_occurrences"][idx_word] = multiple_occ
                dic[sent_id]["annots_per_word_fam_level"][word_fam_level]["stat_sign"][idx_word] = stat_sign
                dic[sent_id]["annots_per_word_fam_level"][word_fam_level]["annots_multiple_occ_min"][idx_word] = d_annots_min
                dic[sent_id]["annots_per_word_fam_level"][word_fam_level]["annots_multiple_occ_max"][idx_word] = d_annots_max

    f_annots.close()
