from Bio import SeqIO
import re
import torch
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel
import os

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print("Using device: {}".format(device))

# Directory path definition.
parent_directory = str(Path(__file__).absolute().parent.parent)
fasta_path = Path("{}/data/initial_data".format(parent_directory))
extracted_dir  = Path("{}/data/extracted_data".format(parent_directory))

# Loading ProtTrans model.
transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))

# Loading the model.
model = T5EncoderModel.from_pretrained(transformer_link)
model = model.to(device)
model = model.eval()

tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)

# Defining sets.
sets = ["training", "testing", "validation"]

# Creating embeddings.
for set in sets:
    sequences = []
    headers = []
    file = fasta_path.joinpath(f'{set}.fasta')

    # Defining the output directory for .pt embedding files.
    output_dir = Path(f'{extracted_dir}/prottrans_{set}_embeddings')

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    # Extracting amino acid sequence for each protein. 
    for seq_record in SeqIO.parse(file, "fasta"):
        headers.append(str(seq_record.id))
        sequences.append(str(seq_record.seq))
    

    for i in range(len(sequences)):
        # This will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        sequences_for_processing = [sequences[i]]
        sequences_for_processing = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences_for_processing]

        # Tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer.batch_encode_plus(sequences_for_processing, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

        required_tokens_index = len(sequences[i])
        amino_acid_representations = embedding_repr.last_hidden_state[0,:required_tokens_index]

        print(f'{headers[i]}')
        print(f'Shape of per-residue embedding of first sequences: {amino_acid_representations.shape}')
        output_file = extracted_dir.joinpath(f'prottrans_{set}_embeddings/{headers[i]}.pt')
        torch.save(amino_acid_representations, output_file)