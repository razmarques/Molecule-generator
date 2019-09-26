"""
This module processes molecular database and formats it to be used in machine learning models
"""

import random
import zinc_downloader as zd
import pandas as pd


def get_random_tranche_id():
    url_dict = zd.build_url_dict()

    code_set = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'}
    random_id = random.sample(code_set, 1) + random.sample(code_set, 1)

    ids_list = url_dict[random_id[0] + random_id[1]]
    full_codes = [i for i in ids_list.keys()]

    random_tranche_id = random.sample(full_codes, 1)
    return random_tranche_id[0]


def sample_n_ids(n):

    ids_set = set()
    while ids_set.__len__() <= n - 1:
        ids_set.add(get_random_tranche_id())

    return list(ids_set)


def build_data_frame(tranche_ids_set):

    header = []
    data = []

    for tranche_id in tranche_ids_set:
        tranche_id_list = zd.get_tranche_id_data(tranche_id, save_file=True)

        if len(header) == 0:
            header = tranche_id_list[0].strip().split("\t")

        for line in tranche_id_list[1:]:
            if len(line) == 0:
                continue
            else:
                data.append(line.strip().split("\t"))

    print("Finished building DataFrame!")

    return pd.DataFrame(data, columns=header)


def build_smiles_dictionary(zinc_df):
    # TODO: improve dictionary to include chemical element strings with size 2 (e.g. Cl, Br)

    character_set = set()
    character_dict = {}

    # Create a set of all individual SMILES characters
    for smiles in zinc_df["smiles"]:

        for char in smiles:
            character_set.add(char)

    # Build a dictionary with SMILES characters with an index value each
    count = 1
    for unique_char in character_set:
        character_dict[unique_char] = count

        # Add the reversed key:value pair to the dictionary. This way the dictionary can both be used to get the index
        # from  a SMILES character or the SMILES character from an index
        character_dict[count] = unique_char

        # iterate count
        count += 1

    return character_dict
