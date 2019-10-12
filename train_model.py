"""
Script file used to train an RNN architecture to learn and predict molecular structures from SMILES data
"""

import format_data as formd
from nn_architects import utilities as utils

# Load database
tranche_id_list = ["AAAA"]
# tranche_id_list = ["AAAA", "GGAB", "JEAA"] # a sample set of tranche IDs, each containing a set of SMILES
smiles_df = formd.build_data_frame(tranche_id_list) # join all SMILES in a single data frame

# Retrieve SMILES data
smi_dict = formd.build_smiles_dictionary(smiles_df, eos_token="\n")

# Build input data array
X, Y = formd.build_input_data(smiles_df, smi_dict)
