#lzma for handling XZ files
#tqdm for displaying a progress bar in the terminal
import os
import lzma
from tqdm import tqdm

#returns a list of all xz filenames in that directory
#os.listdir to get all the file names
#os.path.isfile to check if each one is a file and not a directory or link
#therefore if a file name ends with .xz and is a file it is appended to files list
def xz_files_in_dir(directory):
    files = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "X:\LLM\llm-course\openwebtext" #where xz files are located, ensure a forward slash or double backslash
output_file_train = "output_train.txt" #used for no. of output filenames
output_file_val = "output_val.txt"
vocab_file = "vocab_txt" #saves the vocabulary, new and different characters pushed here

#getting list of filenames and storing in a variable(files)
files = xz_files_in_dir(folder_path)
#counting the total no. of xz files aka the length of our filenames
total_files = len(files)

#calculating the split indices
split_index = int(total_files * 0.9) #90% for training
#process the files for training and validation seperately
files_train = files[:split_index]
files_val = files[split_index:]

#set is a collection of unique characters therefore suitable for holding varied chars in vocabulary
vocab = set()

#PROCESSING WORKFLOW:
#processing .xz files, 
#for each output file we'll process max count files
#for each file we'll open it, read it, and write it's content to the current output file
#and add any unique char to our vocab set
#after processing maxcount files, remove em from our list

# Process the training files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# Process the validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# Write the vocabulary to vocab.txt
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')