"""
This module processes molecular database and formats it to be used in machine learning models
"""

import random
import zinc_downloader as zd
import pandas as pd
import numpy as np


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


def build_input_data(smiles_df, characters_dict):

    eos_token = characters_dict[0]
    n_x = int(characters_dict.__len__() / 2)
    m = smiles_df.shape[0]
    Tx_max = max([len(ch) for ch in smiles_df["smiles"]]) + 1 # added one unit to accommodate the EOS token

    # Print data statistics
    print("Number of training examples: {0}".format(m))
    print("Number of SMILES characters: {0}".format(n_x))
    print("Maximum SMILES length: {0}".format(Tx_max))

    # Initialize data array
    X = np.zeros((n_x, m, Tx_max))
    Y = np.zeros((n_x, m, Tx_max)) # the Y matrix is the data in X shifted on time-step further since we want the RNN to predict the next character in the sequence.

    # Fill X with one-hot vectors for each SMILES training example
    # NOTE: the x one-hot vector for the first time-step is set to the zero.
    for iex in range(m):
        # Extract and canonize the ith SMILES
        smi = smiles_df.loc[iex, "smiles"]
        # smi = Chem.CanonSmiles(smiles_df.loc[iex, "smiles"] + eos_token) # This takes too long. Disabled for the time being

        for t in range(len(smi)):
            ichar = characters_dict[smi[t]]
            X[ichar, iex, t+1] = 1
            Y[ichar, iex, t] = 1

            # If this is the last character, add the EOS character to t+1
            if t == len(smi):
                ieos = characters_dict[eos_token]
                Y[ieos, iex, t+1] = 1

    return X, Y


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

    return pd.DataFrame(data, columns=header)


def build_smiles_dictionary(zinc_df, eos_token):
    # NOTE: dictionary can be improved by defining all character sets in advance

    character_set = set()

    # Initialize character index dictionary with EOS token
    character_dict = {0: eos_token, eos_token: 0}

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
