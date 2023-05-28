import torch
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as PDF
import seaborn as sns
import numpy as np
from scipy import stats
import os

# Defining parent directory.
parent_directory = str(Path(__file__).absolute().parent.parent)

# This function is used to remove any redundant files
# before starting the set extraction.
def removeArrays(set, output_dir):
    try:
        output_dir.joinpath(f'global_{set}_embeddings.npy').unlink()
        output_dir.joinpath(f'{set}_binding_class.npy').unlink()
        output_dir.joinpath(f'global_{set}_SASA.npy').unlink()
    except FileNotFoundError:
        pass

# Function for generating combined embedding and metadata files.
def generate_sets(set, model, output_dir):
    # Defining the directory paths.
    embedding_directory = Path("{}/data/extracted_data/{}_{}_embeddings".format(parent_directory, model, set))
    protein_metadata_directory = Path("{}/data/extracted_data/binary_{}_set".format(parent_directory, set))

    # Output file directory.
    embedding_files = list(embedding_directory.glob('*.pt'))
    file_count = len(embedding_files)
    embbedding_file_regex = re.compile(r'^.+/data/.+?_embeddings/(.+?)\.pt$')

    global_SASA = []
    global_binding_class = []
    j = 1
    global_embeddings = 0
    if (model == 'esm'):
        global_embeddings = np.empty(1280)
    elif (model == 'prottrans'):
        global_embeddings = np.empty(1024)
    else:
        raise ValueError("Invalid value model provided. It can be either 'protrans' or 'esm'.")

    for e_file in embedding_files:
        protein_id = embbedding_file_regex.search(str(e_file)).group(1)
        # This takes the interval of the tensor file by dropping the first
        # and last element, because they are beggining-of-sentence and
        # end-of-sentence tokens.
        if (model == 'esm'):
            embedding = torch.load(e_file)[1:-1].numpy()
        else:
            embedding = torch.load(e_file).numpy()
        metadata_file = protein_metadata_directory.joinpath("{}.dat".format(protein_id))

        with open(metadata_file) as m_file:
            lines = [line.strip() for line in m_file][1:]
        
        for i in range(len(lines)):
            if (lines[i] == ">each_residue_solvent_accessible_surface_area"):
                SASA = lines[i+1].split(',')
                SASA = [float(s) for s in SASA]
                global_SASA = global_SASA + SASA
            if (lines[i] == ">each_residue_binding_class"):
                binding_class = lines[i+1].split(',')
                binding_class = [int(c) for c in binding_class]
                global_binding_class = global_binding_class + binding_class
        print(f'Processing {j}/{file_count} file, before saving {set} data.')
        global_embeddings = np.vstack([global_embeddings, embedding])
        j = j + 1

    np.save(output_dir.joinpath(f'global_{set}_embeddings.npy'), global_embeddings[1:])
    np.save(output_dir.joinpath(f'{set}_binding_class.npy'), global_binding_class)
    np.save(output_dir.joinpath(f'global_{set}_SASA.npy'), global_SASA)


def create_training_set(model='esm'):
    output_dir = Path("{}/data/extracted_data/{}_arrays".format(parent_directory, "training"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = Path("{}/data/final_data/training_X.pt".format(parent_directory))
    if (not output_file.exists()):
        removeArrays("training", output_dir)
        generate_sets("training", model, output_dir)
        generate_normalized_tensor("training", model, output_dir)


def create_testing_set(model='esm'):
    output_dir = Path("{}/data/extracted_data/{}_arrays".format(parent_directory, "testing"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = Path("{}/data/final_data/testing_X.pt".format(parent_directory))
    if (not output_file.exists()):
        removeArrays("testing", output_dir)
        generate_sets("testing", model, output_dir)
        generate_normalized_tensor("testing", model, output_dir)


def create_validation_set(model='esm'):
    output_dir = Path("{}/data/extracted_data/{}_arrays".format(parent_directory, "validation"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = Path("{}/data/final_data/validation_X.pt".format(parent_directory))
    if (not output_file.exists()):
        removeArrays("validation", output_dir)
        generate_sets("validation", model, output_dir)
        generate_normalized_tensor("validation", model, output_dir)


# z-score transformation.
def normalize(column_array, pdf_document, column_number, SASA_bool=False):
    # Generating z-score transformed values.
    if (SASA_bool):
        column_without_zeros = [x for x in column_array if x != 0]
        normalized_without_zeros = stats.zscore(column_without_zeros)
        mean = np.mean(column_without_zeros)
        stdev = np.std(column_without_zeros)
        normalized_column = []
        for i in column_array:
            if i != 0:
                value = (i - mean) / stdev
                normalized_column.append(value)
            else:
                normalized_column.append(i)

        sns.kdeplot(normalized_without_zeros)
    else:
        normalized_column = stats.zscore(column_array)
        sns.kdeplot(normalized_column)
    
    plt.xlabel('Z-score')
    plt.ylabel('Frequency')
    if (SASA_bool):
        plt.title(f'Distribution of normalized SASA values')
    else:
        plt.title(f'Distribution of embedding column {column_number}')
    pdf_document.savefig()
    # Clear plot
    plt.clf()
    
    return normalized_column


# This function normalizes the tensors across
# the columns, this normalizes the embedding
# values and the SASA.
def generate_normalized_tensor(set, model, output_dir):
    print("Normalizing...")
    # Creating PDF file for histogram representation
    # for the dataset.
    pdf_document = PDF("{}/data/final_data/{}_set_histograms.pdf".format(parent_directory, set))

    embeddings = np.load(output_dir.joinpath(f'global_{set}_embeddings.npy'))
    binding_class = np.load(output_dir.joinpath(f'{set}_binding_class.npy'))
    SASA = np.load(output_dir.joinpath(f'global_{set}_SASA.npy'))

    column_num = 1 + embeddings.shape[1]
    row_num = len(SASA)
    X_tensor = np.empty((row_num, column_num))

    normalized_SASA = normalize(SASA, pdf_document, 0, SASA_bool=True)
    X_tensor[:, 0] = normalized_SASA
    
    for i in range(embeddings.shape[1]):
      normalized_embedding = normalize(embeddings[:, i], pdf_document, i+1)
      X_tensor[:, i+1] = normalized_embedding
      print(f'Normalizing {i+1}/1281 column for the {set} data.')

    pdf_document.close()

    X_path = Path("{}/data/final_data/{}_{}_X.pt".format(parent_directory, model, set))
    y_path = Path("{}/data/final_data/{}_{}_y.pt".format(parent_directory, model, set))
    y_tensor = torch.from_numpy(binding_class).reshape(-1, 1).float()

    torch.save(torch.from_numpy(X_tensor), X_path)
    torch.save(y_tensor, y_path)

# Models can be either from ESM ('esm') or ProTrans ('prottrans').
model = 'esm'

create_testing_set(model)
create_validation_set(model)
create_training_set(model)

print("Done.")