import py3Dmol
import boto3
import pandas as pd
import numpy as np


def load_s3_pdb(result_name, model_name = 'model_1_multimer_v3_pred_0'):
    bucket = 'velia-af2-dev'
    prefix = f"outputs/{result_name}/{result_name}.{model_name}.unrelaxed.pdb"
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket, Key=prefix)
    except:
        print(prefix)
        return None
    content = response['Body'].read().decode('utf-8')
    return content

def gradient_color(seq_length):
    colors = []
    for i in range(seq_length):
        fraction = i / (seq_length - 1)
        red = int(255 * (1 - fraction))
        blue = int(255 * fraction)
        colors.append(f"rgb({red},0,{blue})")
    return colors

def view_pdb(pdb, style= 'chain'):
    atoms = pdb_to_atom_coordinates(pdb)
    termini = atoms[(atoms['chain']=='B') & (atoms['atom_name']=='C')].sort_values('residue_index')
    view = py3Dmol.view(width=400, height=700)
    view.addModelsAsFrames(pdb)
    # Set the style and color by chain
    # view.setStyle({'chain':'A'}, {'cartoon': {'color': 'blue'}})
    if style == 'byResidue':
        # Get residues in chain B
        atoms = view.getModel().selectedAtoms()
        chain_b_residues = sorted({atom['resi'] for atom in atoms if atom['chain'] == 'B'})
        # Sequence length for chain B
        sequence_length = len(chain_b_residues)
        # Apply the gradient color to chain B only
        colors = gradient_color(sequence_length)
        for i, resi in enumerate(chain_b_residues):
            view.setStyle({'chain': 'B', 'resi': [str(resi)]}, {'cartoon': {'color': colors[i]}})

    else:
        view.setStyle({'chain':'B'}, {'cartoon': {'color': 'yellow'}})
    view.setStyle({'chain':'C'}, {'cartoon': {'color': 'blue'}})
    view.addLabel('C-terminus', {'position': {'resi': str(termini.iloc[0]['residue_index']), 'chain': 'B', 'atom': 'C'}, 
                             'backgroundColor': 'lightpink', 'fontColor': 'black', 'fontSize': 10})
    view.zoomTo()
    view.show()

def pdb_to_atom_coordinates(pdb_data):
    atoms = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            atom = {
                'chain': line[21],
                'residue': line[17:20].strip(),
                'residue_index': int(line[22:26].strip()),
                'atom_name': line[12:16].strip(),
                'x': float(line[30:38].strip()),
                'y': float(line[38:46].strip()),
                'z': float(line[46:54].strip()),
            }
            atoms.append(atom)
    return pd.DataFrame(atoms)

# Function to calculate Euclidean distance
def calculate_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def coordinates_of_termini(atoms, reference_chain = 'C', termini_chain = 'B'):
    # Find N-terminus and C-terminus of chain A
    termini_chain_atoms = atoms[(atoms['chain']==termini_chain) & (atoms['atom_name']=='CA')].sort_values('residue_index')
    n_terminus_a = termini_chain_atoms.iloc[0]
    n_terminus_a = (n_terminus_a['x'], n_terminus_a['y'], n_terminus_a['z'])
    c_terminus_a = termini_chain_atoms.iloc[-1]
    c_terminus_a = (c_terminus_a['x'], c_terminus_a['y'], c_terminus_a['z'])
    return n_terminus_a, c_terminus_a

def calculate_chain_distance_to_termini(atoms, reference_chain = 'C', termini_chain = 'B'):
    # Calculate distances from each residue in chain B to N-terminus and C-terminus of chain A
    n_terminus, c_terminus = coordinates_of_termini(atoms, termini_chain)
    distances = []
    for ix, atom in atoms.iterrows():
        if atom['chain'] == reference_chain and atom['atom_name'] == 'CA':
            res_coord = (atom['x'], atom['y'], atom['z'])
            distance_to_n_terminus = calculate_distance(res_coord, n_terminus)
            distance_to_c_terminus = calculate_distance(res_coord, c_terminus)
            distances.append({
                'residue': atom['residue'],
                'residue_index': atom['residue_index'],
                'distance_to_n_terminus': distance_to_n_terminus,
                'distance_to_c_terminus': distance_to_c_terminus
            })
    return pd.DataFrame(distances)