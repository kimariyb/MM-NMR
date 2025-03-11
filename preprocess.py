import os

from tqdm import tqdm
from rdkit import Chem, RDLogger
from utils.smiles import split_smiles
from utils.vocab import WordVocab   

RDLogger.DisableLog('rdApp.*')


def generate_corpus(data_path, save_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data file not found: {data_path}')
    if os.path.exists(save_path):
        return True
    
    suppl = Chem.SDMolSupplier(data_path, removeHs=False, sanitize=True)
    smiles_list = []
    
    for mol in tqdm(suppl, desc='Generating corpus'):
        if mol is None:
            continue
        
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        smiles_list.append(smiles)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for smiles in tqdm(smiles_list, desc='Writing corpus'):
            f.write(split_smiles(smiles) + '\n')

    return False


def generate_vocab(corpus_path, save_path, vocab_size=None, min_freq=1):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f'Corpus file not found: {corpus_path}')
    if os.path.exists(save_path):
        return True
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        vocab = WordVocab(f, max_size=vocab_size, min_freq=min_freq)
    
    vocab.save_vocab(save_path)
    return False


if __name__ == '__main__':
    # First, generate the corpus file
    if generate_corpus('./data/carbon/raw/carbon_dataset.sdf', './data/carbon/processed/corpus.txt'):
        print('Corpus file already exists. Skipping corpus generation.')
        # Then, preprocess the corpus file
        if generate_vocab('./data/carbon/processed/corpus.txt', './data/carbon/processed/vocab.pkl', vocab_size=None, min_freq=1):
            print('Vocab file already exists. Skipping vocab generation.')
        else:
            print('Vocab file generated successfully.')
    else:
        print('Corpus file generated successfully.')
        


    # 