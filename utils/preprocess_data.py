import re
import argparse

from rdkit import Chem, RDLogger
from tqdm import tqdm


RDLogger.DisableLog('rdApp.*')


# Path to the sdf file containing the spectra
DATASET_PATH = '/workspace/MM-NMR/data/dataset/nmrshiftdb2withsignals.sd'


def clean_dataset(suppl):
    r"""
    Cleans a list of RDKit molecules by removing invalid molecules and molecules with more than 40 atoms.

    Parameters
    ----------
    suppl : list of Chem.rdchem.Mol
        The list of RDKit molecules to clean.

    Returns
    -------
    list of Chem.rdchem.Mol
        The cleaned list of RDKit molecules.
    """
    cleaned_suppl = []
    
    for mol in tqdm(suppl, desc="Cleaning dataset", total=len(suppl)):
        # Check if the molecule is valid
        if mol is None:
            continue
        
        # Only consider molecules with less than 40 atoms
        if mol.GetNumAtoms() >= 40:
            continue
        
        # Only consider molecules with H, C, N, O, F, Si, P, S, Cl, Br, and I atoms
        if not all(atom.GetAtomicNum() in [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53] for atom in mol.GetAtoms()):
            continue
        
        # Remove the ionic
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if '+' in smiles or '-' in smiles:
            continue
        
        cleaned_suppl.append(mol)
        
    return cleaned_suppl


def export_file(suppl: list, output_path: str):
    r"""
    Exports a list of RDKit molecules to a file.

    Parameters
    ----------
    suppl : list of Chem.rdchem.Mol
        The list of RDKit molecules to export.
    output_path : str
        The path to the output file.
    """
    writer = Chem.SDWriter(output_path)
    
    for mol in suppl:
        props = mol.GetPropsAsDict()
        writer.SetProps(list(props.keys()))  
        
        writer.write(mol)
        
    writer.close()
   

def create_hydrogen_dataset(data_path: str):
    suppl = Chem.SDMolSupplier(data_path, removeHs=False, sanitize=True)

    dataset = []
    
    cleaned_suppl = clean_dataset(suppl)
        
    for mol in tqdm(cleaned_suppl, desc="Validating dataset", total=len(cleaned_suppl)):        
        # get the properties of the molecule
        prop_names = mol.GetPropNames(includePrivate=False, includeComputed=False)
        has_spectrum = False
        
        for prop in prop_names:
            pattern = r"^Spectrum 1H \d+$"
            if bool(re.match(pattern, prop)):
                has_spectrum = True
                break
        
        if has_spectrum:
            dataset.append(mol)
        
    print(f"Valid molecules found: {len(dataset)}")
    
    export_file(dataset, output_path='hydrogen_dataset.sdf')
        

def create_carbon_dataset(data_path: str):
    r"""
    Creates a dataset of carbon spectra from a given sdf file.
    
    Parameters
    ----------
    data_path : str
        The path to the sdf file containing the carbon spectra.
    """
    suppl = Chem.SDMolSupplier(data_path, removeHs=False, sanitize=True)

    dataset = []
    cleaned_suppl = clean_dataset(suppl)
        
    for mol in tqdm(cleaned_suppl, desc="Validating dataset", total=len(cleaned_suppl)):        
        # get the properties of the molecule
        prop_names = mol.GetPropNames(includePrivate=False, includeComputed=False)
        # check if the molecule has a carbon spectrum
        for prop in prop_names:
            pattern = r"^Spectrum 13C \d+$"
            if bool(re.match(pattern, prop)):
                dataset.append(mol)
                break
            else:
                continue
        
    print(f"Valid molecules found: {len(dataset)}")
        
    # export the dataset to a sdf file
    export_file(dataset, output_path='carbon_dataset.sdf')


def create_dataset(data_path: str, element: str = 'carbon'):
    r"""
    Creates a dataset of spectra from a given sdf file.
    
    Parameters
    ----------
    data_path : str
        The path to the sdf file containing the spectra.
    element : str
        The element to create the dataset for.
        Options: 'carbon', 'hydrogen', 'fluorine'
    """
    if element == 'carbon':
        create_carbon_dataset(data_path=data_path)
    elif element == 'hydrogen':
        create_hydrogen_dataset(data_path=data_path)
    else:
        raise ValueError("Invalid element specified. Options: 'carbon', 'hydrogen'")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        '--element', 
        '-e', 
        type=str, 
        default='carbon', 
        required=True, 
        help='The element to create the dataset for. Options: "carbon", "hydrogen", "fluorine"'
    )
    
    args = args.parse_args()
     
    create_dataset(data_path=DATASET_PATH, element=args.element)