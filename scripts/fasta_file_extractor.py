# This program extracts the protein sequences from each
# of the 3 data sets and stores them as 3 seperate fasta
# format files.

from pathlib import Path
from Bio.SeqUtils import seq1
import re

# Defining the directory paths.
parent_directory = str(Path(__file__).absolute().parent.parent)

initial_data_directories = [Path(parent_directory + '/data/initial_data/training_set'), 
                      Path(parent_directory + '/data/initial_data/testing_set'), 
                      Path(parent_directory + '/data/initial_data/validation_set')]

# Defining regex patterns.
data_output_filename_regex = re.compile(r'^.+/data/initial_data/(.+?)_set$')
file_protein_id_regex = re.compile(r'^.+?/(.{4})_\w+?\.full_graph\.vertices\.csv$')
atom_id_regex = re.compile(r'c<\w{1}>r<(-?\d+)>.*?R<(\w{3})>\w{1}<.+?>')

# Defining global variables.
previous_residue_number = 0
first_line_identifier = 1
residue_array = []

# Main protein extraction loop.
# First loop iterates over the each set directory.
for directory in initial_data_directories:
    filename_match = data_output_filename_regex.search(str(directory))
    data_output_file = open("{}/data/initial_data/{}.fasta".format(parent_directory, filename_match.group(1)), "w")
    files_in_directory = directory.glob('*.vertices.csv')
    
    # Loop iterates over files in one of the 3 directories.
    for file in files_in_directory:
        protein_id_match = file_protein_id_regex.search(str(file))
        data_output_file.write(">{}\n".format(protein_id_match.group(1)))

        with open(file) as read_file:
            lines = [line.strip() for line in read_file][1:]
        lines.append('EOF')

        # Loop iterates over each line from a single file.
        for line in lines:
            # This function is used to save 3-letter residue name into
            # an array which is used at the end of the loop together
            # with seq1() function to extract the 1-letter aminoacid
            # representations.
            def save():
                global previous_residue_number

                previous_residue_number = current_residue_number
                residue_array.append(str(current_residue))

            # Breaks if end of file.
            if (line == 'EOF'):
                save()
                break
        
            columns = line.split(',')
            regex_match = atom_id_regex.search(columns[0])
            current_residue_number = regex_match.group(1)
        
            if (current_residue_number != previous_residue_number and first_line_identifier == 0): 
                save()

            # Used to specify when first line is encountered while reading
            # the file.
            if first_line_identifier == 1:
                previous_residue_number = current_residue_number
                first_line_identifier = 0
        
            current_residue = regex_match.group(2)
        
        single_letter_sequence = seq1(''.join(residue_array))
        data_output_file.write("{}\n".format(single_letter_sequence))

        previous_residue_number = 0
        first_line_identifier = 1
        residue_array = []

        print(file)

    data_output_file.close()