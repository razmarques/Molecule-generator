"""
This module handles the download of molecules from the ZINC database.
A list of ZINC URLS accessing the database is available in the file ZINC-downloader-2D-txt.uri

Author: Rodolfo Marques
"""

import os
import requests
from collections import defaultdict


def build_url_dict(urls_file="ZINC-downloader-2D-txt.uri"):

    with open(urls_file, "r") as url_file:
        urls_list = url_file.read().strip().split("\n")

    # Retrieve a list of tranche codes and store them in a dictionary
    tranche_urls = defaultdict(dict)
    for url in urls_list:
        code_id = url.replace("http://files.docking.org/2D/", "").replace(".txt", "")
        split_code_id = code_id.split("/")

        # Update tranche_urls dictionary to return a tranche URL given a tranche_id
        tranche_urls[split_code_id[0]][split_code_id[1]] = url
        tranche_urls[split_code_id[1]] = url

    return tranche_urls


def download_tranche_data(tranche_id):
    print("Downloading {0}...".format(tranche_id))

    tranche_urls = build_url_dict()

    if tranche_urls.get(tranche_id) is None:
        print("Incorrect tranche ID")
    else:
        url = tranche_urls[tranche_id]
        url_req = requests.get(url)
        return url_req.text


def get_tranche_id_data(tranche_id, save_file=True):

    # Set up directory for zinc files
    zinc_files_dir = "./data/zinc_files/"
    if not os.path.isdir(zinc_files_dir):
        os.mkdir(zinc_files_dir)
    zinc_file = zinc_files_dir + tranche_id + ".dat"

    # Check if tranche id zinc file is in zinc_files_dir
    if os.path.isfile(zinc_file):
        with open(zinc_file, "r") as file:
            content = file.read()
    else:
        urls_list = build_url_dict()
        content = download_tranche_data(tranche_id)

    # Save file if save flag is enabled
    if save_file:
        with open(zinc_file, "w") as zinc_file:
            zinc_file.write(content)

    return content.strip().split("\n")
