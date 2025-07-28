import os
import torch
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from loaders.utils import mol2data
from collections import defaultdict
from sklearn.utils import shuffle


class CarbonDataset(InMemoryDataset):
    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.max_num_conformers = max_num_conformers
        super(CarbonDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0]) 
 
    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")
    
    @property
    def raw_file_names(self):
        return 'carbon_dataset.sdf'

    @property
    def processed_file_names(self):
        return (
            'processed_data.pt'
            if self.max_num_conformers is None
            else f'processed_data_{self.max_num_conformers}.pt'
        )
    
    @property
    def num_molecules(self):
        return len(self.data)
    
    def process(self):
        data_list = []
        suppl = Chem.SDMolSupplier(
            os.path.join(self.raw_dir, self.raw_file_names), removeHs=False
        )

        mols = defaultdict(list)
        raw_sdf_file = os.path.join(self.raw_dir, self.raw_file_names)

        with tqdm(
            Chem.SDMolSupplier(raw_sdf_file, removeHs=False),
            desc="Processing carbon dataset",
        ) as suppl:
            for idx, mol in enumerate(suppl):
                if mol is None:
                    continue

                # get the id for the molecule
                mol_id = mol.GetProp("nmrshiftdb2 ID")
                mol_name = mol.GetProp("_Name")
                mol_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

                data = mol2data(mol)
                data.name = mol_name
                data.id = mol_id
                data.smiles = mol_smiles

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                mols[mol_name].append(data)


            # extract spectrum
            spectrum = self.extract_spectrum(mol)
            if spectrum is None:
                continue

            # get the carbon atoms
            for i, atom in enumerate(mol.GetAtoms()):
                if i in spectrum:
                    atom.SetProp('shift', str(spectrum[i]))
                    atom.SetBoolProp('mask', True)
                else:
                    atom.SetProp('shift', str(0))
                    atom.SetBoolProp('mask', False)

            # create the shift tensor and mask tensor
            shift_tensor = torch.zeros(mol.GetNumAtoms(), 1)
            mask_tensor = torch.zeros(mol.GetNumAtoms(), 1)
                

            
            data_list.append(data)
            
        torch.save(self.collate(data_list), self.processed_paths[0])

    
    def extract_spectrum(self, mol):
        r"""
        Extract spectrum from sdf file.
        
        Example of sdf file:
        
        > <Spectrum 13C 0>
        17.6;0.0Q;10|18.3;0.0T;0|22.6;0.0Q;12|26.5;0.0T;6|31.7;0.0T;5|33.5;0.0S;2|33.5;0.0S;14|41.8;0.0T;1|
        42.0;0.0S;4|42.2;0.0D;3|78.34;0.0S;9|140.99;0.0S;8|158.3;0.0D;7|193.4;0.0D;15|203.0;0.0D;11|
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
        
        Returns
        -------
        spectrum : dict
            Dictionary containing spectrum information.
        """
        mol_props = mol.GetPropsAsDict()
        atom_shifts = {}

        for key in mol_props.keys():
            if key.startswith('Spectrum 13C'):
                for shift in mol_props[key].split('|')[:-1]:
                    [shift_value, _, shift_idx] = shift.split(';')
                    shift_value, shift_idx = float(shift_value), int(shift_idx)
                    if shift_idx not in atom_shifts: 
                        atom_shifts[shift_idx] = []
                        
                    atom_shifts[shift_idx].append(shift_value)
                    
        for j in range(mol.GetNumAtoms()):
            if j in atom_shifts:
                atom_shifts[j] = np.median(atom_shifts[j])
        
        return atom_shifts


    def get_idx_split(self, train_ratio=0.8, valid_ratio=0.1, seed=123):
        molecule_ids = shuffle(range(self.num_molecules), random_state=seed)
        train_size = int(train_ratio * self.num_molecules)
        valid_size = int(valid_ratio * self.num_molecules)

        train_idx = torch.tensor(molecule_ids[:train_size])
        valid_idx = torch.tensor(molecule_ids[train_size : train_size + valid_size])
        test_idx = torch.tensor(molecule_ids[train_size + valid_size :])

        return train_idx, valid_idx, test_idx
