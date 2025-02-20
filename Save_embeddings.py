import os
import re
import numpy as np
import pandas as pd
import argparse
import logging
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor

import glob
from natsort import natsorted

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_chunk", type=str, required=True, help="Input file containing sequences (train, val, test); name format: 'chunk_1.tsv'")
    parser.add_argument("-model", type=str, help="The directory of pre-trained model")
    parser.add_argument("-output_folder", type=str, help="The directory of output")
    parser.add_argument("-device", type=str, default="cuda:0", help="The device to run the model")
    parser.add_argument("-batchSize", type=int, default=128, help="The batch size for the model")
    parser.add_argument("-tokenIdx", type=int, default=255, help="The index of the nucleotide")
    parser.add_argument("-dataset_type", type=str, help="Specify the partition (train, val, test) to save in the NPZ files")
    return parser.parse_args()

class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
        input_ids = encoding['input_ids']
        return {
            'sequence': sequence,
            'input_ids': input_ids.squeeze()
        }
    
def extract_chunk_id(filename):
    # Use regular expression to match 'chunk_' followed by any characters and then digits, ending with '.tsv'
    match = re.search(r'chunk_.*?(\d+)\.tsv', filename)
    if match:
        return int(match.group(1))  # Extract the chunk ID as an integer
    else:
        raise ValueError("Filename does not match the expected format.")
    
def load_data(filepath):
    logging.info(f"Loading data from {filepath}")
    data = pd.read_csv(filepath, delimiter='\t')
    return data['sequences'].tolist(), data['label'].tolist()

def create_dataloader(sequences, tokenizer, batch_size):
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = SequenceDataset(sequences, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def load_model_and_tokenizer(model_dir, device):
    logging.info(f"Loading model and tokenizer from {model_dir}")
    
    # Determine the appropriate dtype based on the GPU capabilities
    def get_optimal_dtype():
        if not torch.cuda.is_available():
            logging.info("Using float32 as no GPU is available.")
            return torch.float32  

        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)

        if capability[0] >= 8:  # sm_80 or higher
            logging.info("Using bfloat16 as the GPU supports sm_80 or higher.")
            return torch.bfloat16
        elif capability[0] >= 6:  # sm_60 or higher
            logging.info("Using float16 as the GPU supports sm_60 or higher.")
            return torch.float16
        else:
            logging.info("Using float32 as the GPU does not support float16 or bfloat16.")
            return torch.float32

    optimal_dtype = get_optimal_dtype()

    # Load the model with the selected dtype
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=optimal_dtype)
    except Exception as e:
        logging.error(f"Failed to load model with {optimal_dtype}, falling back to float32. Error: {e}")
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer

def extract_embeddings(model, dataloader, device, tokenIdx):
    logging.info("Extracting embeddings")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, output_hidden_states=True)
            token_embedding = outputs.hidden_states[-1][:, tokenIdx, :].to(torch.float32).cpu().numpy()
            embeddings.append(token_embedding)
    embeddings = np.concatenate(embeddings, axis=0)
    # average forward and reverse embeddings
    hidden_size = embeddings.shape[-1] // 2
    forward = embeddings[..., 0:hidden_size]
    reverse = embeddings[..., hidden_size:]
    reverse = reverse[..., ::-1]
    averaged_embeddings = (forward + reverse) / 2
    return averaged_embeddings

def combine_embeddings(chunk_dir, dataset_type):
    logging.info("Combining_chunks")
    file_pattern = os.path.join(chunk_dir, f'{dataset_type}_chunk_embeddings_*.npz')
    # We retrieve the list of files. It's important to sort them in a human-readable order to ensure the labels don't get mixed up later.
    file_list = natsorted(glob.glob(file_pattern))
    combined_data = {}
    for file in file_list:
        data = np.load(file)
        for key in data.keys():
            if key not in combined_data:
                combined_data[key] = []
            combined_data[key].append(data[key])
    for key in combined_data:
        combined_data[key] = np.concatenate(combined_data[key], axis=0)
    output_file = os.path.join(chunk_dir, "combined_embeddings.npz")
    np.savez(output_file, **combined_data)
    logging.info(f"Combined embeddings were saved as : {output_file}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    
    chunk_sequences, chunk_labels = load_data(args.input_chunk)
    chunk_loader = create_dataloader(chunk_sequences, tokenizer, args.batchSize)
    chunk_embeddings = extract_embeddings(model, chunk_loader, args.device, args.tokenIdx)
    logging.info(f"Saving embeddings to {args.output_folder}")
    chunk_id = extract_chunk_id(args.input_chunk)
    # Save embeddings to NPZ file with dynamic keys
    np.savez_compressed(
        os.path.join(args.output_folder, f'{args.dataset_type}_chunk_embeddings_{chunk_id}.npz'),
        **{args.dataset_type: chunk_embeddings}
    )
    combine_embeddings(args.output_folder, args.dataset_type)

if __name__ == "__main__":
    main()