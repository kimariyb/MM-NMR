import re

from rdkit import Chem, RDLogger
from tqdm import tqdm

from loader.Carbon import CarbonSpectraDataset
from loader.Hydrogen import HydrogenSpectraDataset
from loader.Fluorine import FluorineShiftDataset


RDLogger.DisableLog('rdApp.*')


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
    suppl = collect(data_path)

    dataset = []
        
    for mol in tqdm(suppl, desc="Validating dataset", total=len(suppl)):
        # Check if the molecule is valid
        if mol is None:
            print(f"Invalid molecule found in {data_path}")
            continue
        
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
    suppl = collect(data_path)

    dataset = []
        
    for mol in tqdm(suppl, desc="Validating dataset", total=len(suppl)):
        # Check if the molecule is valid
        if mol is None:
            print(f"Invalid molecule found in {data_path}")
            continue
        
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


def create_fluorine_dataset(data_path: str):
    r"""
    Creates a dataset of fluorine spectra from a given sdf file.
    
    Parameters
    ----------
    data_path : str
        The path to the sdf file containing the fluorin spectra.
    """
    suppl = collect(data_path)
    
    dataset = []
        
    for mol in tqdm(suppl, desc="Validating dataset", total=len(suppl)):
        # Check if the molecule is valid
        if mol is None:
            print(f"Invalid molecule found in {data_path}")
            continue
        
        # get the properties of the molecule
        prop_names = mol.GetPropNames(includePrivate=False, includeComputed=False)
        # check if the molecule has a carbon spectrum
        for prop in prop_names:
            pattern = r"^Spectrum 19F \d+$"
            if bool(re.match(pattern, prop)):
                dataset.append(mol)
                break
            else:
                continue
        
    print(f"Valid molecules found: {len(dataset)}")
        
    # export the dataset to a sdf file
    export_file(dataset, output_path='fluorin_dataset.sdf')


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
    elif element == 'fluorine':
        create_fluorine_dataset(data_path=data_path)
    else:
        raise ValueError("Invalid element specified. Options: 'carbon', 'hydrogen', 'fluorine'")


def collect(data_path: str):
    r"""
    Only H, B, C, O, N, F, Si, P, S, Cl, Br, I are collected.

    Parameters
    ----------
    data_path : str
        The path to the sdf file containing the spectra.

    Returns
    -------
    list of Chem.rdchem.Mol
        The list of selected atoms.
    """
    # Define the set of target elements
    target_elements = {"H", "B", "C", "O", "N", "F", "Si", "P", "S", "Cl", "Br", "I"}

    suppl = Chem.SDMolSupplier(data_path, removeHs=False, sanitize=True)

    # Initialize the list to store valid molecules
    valid_molecules = []

    # Iterate over the molecules in the SDF file
    for mol in tqdm(suppl, desc="Processing molecules", total=len(suppl)):
        # Check if the molecule is valid
        if mol is None:
            print(f"Invalid molecule found in {data_path}")
            continue

        # Get the set of elements in the molecule
        elements_in_mol = {atom.GetSymbol() for atom in mol.GetAtoms()}

        # Check if the molecule contains only the target elements
        if elements_in_mol.issubset(target_elements):
            valid_molecules.append(mol)

    return valid_molecules
    

if __name__ == "__main__":
    # data_path = './data/nmrshiftdb2withsignals.sd'
    # create_dataset(data_path=data_path, element='carbon')
    
    dataset = CarbonSpectraDataset(root='./data')
    data = dataset[0]
    
    print(data.num_nodes)
    print(data.x.shape)
    print(data.edge_attr.shape)
    print(data.fingerprint.shape)
    print(data.vector.shape)
        
    from utils.Features import GetAtomFeaturesDim, GetBondFeaturesDim
    print(GetAtomFeaturesDim())
    print(GetBondFeaturesDim())
    
    from network.Transformer import MultiViewRepresentation
    
    model = MultiViewRepresentation(embed_dim=128)
    out = model(data)
