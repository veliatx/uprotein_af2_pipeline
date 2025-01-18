from smart_open import open
import json
import pandas as pd
import numpy as np

receptor_file = "../code/uniprot_singlePass_transmembrane_receptors_simplified.tsv"
ligand_files = [f"../peptides/{lf}" for lf in [
    "20240812_peptidomics_wave5.json",
    "20240812_round_robin_portfolio.json",
    "20240905_receptor_ligand_prediction_peptidomics_run2.json",
    "20240805_metaorf_minimum_votes.json"]]

receptor_names = []
for row in pd.read_csv(receptor_file, index_col=0).iterrows():
    receptor_names.append(row[1]["Entry Name"])

ligand_names = []
ligands = []
for ligand_file in ligand_files:
    for ligand, coordinates in json.load(open(ligand_file)).items():
        for ligand_start, ligand_end in coordinates:
            ligand_names.append(f"{ligand}_{ligand_start}_{ligand_end}")
            ligands.append([ligand, ligand_start, ligand_end])

ranking_confidence_scores = []
for receptor in receptor_names:
    current_ranking_confidence_scores = []
    for ligand, ligand_start, ligand_end in ligands:
        experiment_name = f"{receptor}_{ligand}_{ligand_start}_{ligand_end}_13982528712754417"
        try:
            with open(f"s3://velia-af2-dev/outputs/{experiment_name}/{experiment_name}.model_1_multimer_v3_pred_0.scores.json") as f:
                ranking_confidence = json.load(f)["ranking_confidence"]
        except:
            ranking_confidence = -np.inf
        current_ranking_confidence_scores.append(ranking_confidence)
    ranking_confidence_scores.append(current_ranking_confidence_scores)

ranking_confidence_df = pd.DataFrame(ranking_confidence_scores, index=receptor_names, columns=ligand_names)

with open("single_pass_scores.txt", "w") as score_file:
    high_conf_pairs = []
    for row in ranking_confidence_df.iterrows():
        receptor = row[0]
        for ligand, confidence_score in row[1].items():
            score_file.write(f"{receptor}_{ligand}_13982528712754417\t{confidence_score}\n")
