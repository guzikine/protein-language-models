# This program is used to extract the amino acid
# embeddings for each dataset.

import torch
from Bio import SeqIO
from pathlib import Path
import os

# Loading the ESM2 model.
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

# Directory path definition.
parent_directory = str(Path(__file__).absolute().parent.parent)
fasta_path = Path("{}/data/initial_data".format(parent_directory))
extracted_dir  = Path("{}/data/extracted_data".format(parent_directory))

# Switching to CUDA if possible.
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)

# Model loading.
batch_converter = alphabet.get_batch_converter()
model.eval()
model.to(device)

# Defining sets.
sets = ["training", "testing", "validation"]

# Creating embeddings.
for set in sets:
    sequences = []
    headers = []
    file = fasta_path.joinpath(f'{set}.fasta')

    # Defining the output directory for .pt embedding files.
    output_dir = Path(f'{extracted_dir}/esm_{set}_embeddings')

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    # Extracting amino acid sequence for each protein. 
    for seq_record in SeqIO.parse(file, "fasta"):
        headers.append(str(seq_record.id))
        sequences.append(str(seq_record.seq))
    
    for i in range(len(sequences)):
        data = [(str(headers[i]), str(sequences[i]))]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
  
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33][0]
        output_file = extracted_dir.joinpath(f'esm_{set}_embeddings/{headers[i]}.pt')
        print(str(output_file))
        torch.save(token_representations, output_file)
