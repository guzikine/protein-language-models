# This program extracts each residue from a protein sequence that 
# is provided in the {protein}.vertices.csv file. Each residue
# has data about the solvent accessible surface area and it's
# binding class, where 0 means that it is not solvent accessible and
# 1 meaning it is. More information in the:
# data/initial_data/POBSUDL.pdf file.

from pathlib import Path
from Bio.SeqUtils import seq1
import re
import os

# Defining the directory paths.
parent_directory = str(Path(__file__).absolute().parent.parent)

# Extracted data directory.
extracted_dir = str(Path(f'{parent_directory}/data/extracted_data'))

if not os.path.exists(extracted_dir):
    os.makedirs(extracted_dir)

initial_data_directories = [Path(parent_directory + '/data/initial_data/training_set'), 
                      Path(parent_directory + '/data/initial_data/testing_set'), 
                      Path(parent_directory + '/data/initial_data/validation_set')]

# Defining regex patterns.
data_output_filename_regex = re.compile(r'^.+/data/initial_data/(.+?)_set$')
file_protein_id_regex = re.compile(r'^.+?/(.{4})_\w+?\.full_graph\.vertices\.csv$')
atom_id_regex = re.compile(r'c<\w{1}>r<(-?\d+)>.*?R<(\w{3})>\w{1}<.+?>')

# Defining global variables.
previous_residue_number = 0
SASA = 0
binding_class = 0
first_line_identifier = 1
residue_array = []
SASA_array = []
binding_class_array = []

# Main protein extraction loop.
# First loop iterates over the each set directory.
for directory in initial_data_directories:
    files_in_directory = directory.glob('*.vertices.csv')
    filename_match = data_output_filename_regex.search(str(directory))
    
    # Defining the output directory for .dat files.
    output_dir = Path(f'{extracted_dir}/binary_{filename_match.group(1)}_set')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop iterates over files in one of the 3 directories.
    for file in files_in_directory:
        protein_id_match = file_protein_id_regex.search(str(file))
        data_output_file = open("{}/data/extracted_data/binary_{}_set/{}.dat".format(parent_directory, filename_match.group(1), protein_id_match.group(1)), "w")

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
                global SASA
                global previous_residue_number
                global binding_class

                previous_residue_number = current_residue_number
                residue_array.append(str(current_residue))
                SASA_array.append(SASA)
                SASA = 0
                binding_class_array.append(0) if binding_class == 0 else binding_class_array.append(1)
                binding_class = 0

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
            SASA = SASA + float(columns[5])
            binding_class = binding_class + int(columns[7])
        
        single_letter_sequence = seq1(''.join(residue_array))
        
        # Writing the main body of the output file.
        data_output_file.write(">protein_id\n{}\n".format(protein_id_match.group(1)))
        data_output_file.write(">protein_sequence\n{}\n".format(single_letter_sequence))
        data_output_file.write(">each_residue_solvent_accessible_surface_area\n{}\n".format(','.join(map(str, SASA_array))))
        data_output_file.write(">each_residue_binding_class\n{}\n".format(','.join(map(str, binding_class_array))))

        previous_residue_number = 0
        SASA = 0
        binding_class = 0
        first_line_identifier = 1
        residue_array = []
        SASA_array = []
        binding_class_array = []

        print(file)
        
    data_output_file.close()