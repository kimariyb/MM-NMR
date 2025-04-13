import os
import torch
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from data.features import mol2graph, mol2geometry


class CarbonDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
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
        return 'processed_data.pt'
    
    def process(self):
        data_list = []
        suppl = Chem.SDMolSupplier(os.path.join(self.raw_dir, self.raw_file_names))

        for mol in tqdm(suppl, desc="Processing carbon dataset", unit="mol", ncols=100, total=len(suppl)):            
            if mol is None:
                continue

            # create graph
            graph = mol2graph(mol)
            if graph is None:
                continue 

            data = Data()
            data.smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) 
            data.inchi = mol.GetProp('INChI key')
            data.x = torch.tensor(graph['node_feat'], dtype=torch.float)
            data.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            data.edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.float)   

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
                
            # get shift and mask tensors
            for i, atom in enumerate(mol.GetAtoms()):
                shift_tensor[i] = float(atom.GetProp('shift'))
                mask_tensor[i] = float(atom.GetBoolProp('mask'))

            data.y = shift_tensor.clone().detach().to(torch.float)
            data.mask = mask_tensor.clone().detach().to(torch.bool).reshape(-1)

            # get geometries
            pos, z = mol2geometry(mol)
            if pos is None or z is None:
                continue
     
            data.pos = torch.tensor(pos, dtype=torch.float)
            data.z = torch.tensor(z, dtype=torch.long)
            
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


    def filter_invalid_inchi(self, invalid_inchi_keys):
        r"""
        Filter invalid inchi keys from dataset.
        
        Parameters
        ----------
        invalid_inchi_keys : list
            List of invalid inchi keys.
        
        Returns
        -------
        new_dataset : CarbonDataset
            New dataset instance with filtered data.
        """
        # print the hint
        print(f"Filtering invalid inchi keys: {invalid_inchi_keys}")

        # 1. load all data
        data_list = [self.get(i) for i in range(len(self))]
        
        # 2. filter invalid inchi keys
        valid_data = [
            data for data in data_list
            if data.inchi not in invalid_inchi_keys
        ]
        
        # 3. create new dataset instance
        new_dataset = CarbonDataset(root=self.root)
        
        # 4. re-compose and save data
        new_dataset._data, new_dataset._slices = self.collate(valid_data)
        torch.save((new_dataset._data, new_dataset._slices), new_dataset.processed_paths[0])
        
        return new_dataset
