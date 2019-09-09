"""
Script file used to train the Deep Neural Network model to a database of SMILES
"""

import numpy as np
from rdkit import Chem

import format_data as formd


# Load database
tranche_ide_list = ["AAAA", "GGAB", "JEAA"]
smiles_df = formd.build_data_frame(tranche_ide_list)

# Retrieve and canonize SMILES data. Retrieve logp data
smi_dict = formd.build_smiles_dictionary(smiles_df)
