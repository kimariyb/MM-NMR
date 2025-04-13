import os
import torch
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from data.features import mol2graph, mol2geometry


class HydrogenDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super(HydrogenDataset, self).__init__(root, transform, pre_transform, pre_filter)       
        self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")
    
    @property
    def raw_file_names(self):
        return 'hydrogen_dataset.sdf'

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
            # sort the spectrum by atom index
            spectrum = {k: v for k, v in sorted(spectrum.items())}
            
            # get the hydrogen atoms
            for i, atom in enumerate(mol.GetAtoms()):
                if i in spectrum:
                    atom.SetProp('shift', str(spectrum[i]))
                    atom.SetBoolProp('mask', True)
                else:
                    atom.SetProp('shift', str([0]))
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
        
        > <Spectrum 1H 3>
        1.43;0.0;11|2.34;0.0;3|1.63;0.0;4|2.41;0.0;0|2.72;0.0;1|1.8;0.0;2|0.99;0.0;14|
        1.15;0.0;14|1.42;0.0;14|1.48;0.0;3|1.48;0.0;12|1.57;0.0;12|0.65;0.0;12|3.41;0.0;9|
        6.24;0.0;6|1.08;0.0;10|1.14;0.0;10|2.09;0.0;10|4.64;0.0;8|5.13;0.0;8|-0.37;0.0;15|
        1.69;0.0;1|2.03;0.0;0|1.87;0.0;4|

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
            if key.startswith('Spectrum 1H'):
                tmp_dict = {}

                for shift in mol_props[key].split('|')[:-1]:
                    [shift_val, _, shift_idx] = shift.split(';')
                    shift_val, shift_idx = float(shift_val), int(shift_idx)
                    if shift_idx not in atom_shifts: 
                        atom_shifts[shift_idx] = []
                    if shift_idx not in tmp_dict: 
                        tmp_dict[shift_idx] = []

                    tmp_dict[shift_idx].append(shift_val)

                for shift_idx in tmp_dict.keys():
                    atom_shifts[shift_idx].append(tmp_dict[shift_idx])

        for shift_idx in atom_shifts.keys():
            max_len = np.max([len(shifts) for shifts in atom_shifts[shift_idx]])
            for i in range(len(atom_shifts[shift_idx])):
                if len(atom_shifts[shift_idx][i]) < max_len:
                    if len(atom_shifts[shift_idx][i]) == 1:
                        atom_shifts[shift_idx][i] = [atom_shifts[shift_idx][i][0] for _ in range(max_len)]
                    elif len(atom_shifts[shift_idx][i]) > 1:
                        while len(atom_shifts[shift_idx][i]) < max_len:
                            atom_shifts[shift_idx][i].append(np.mean(atom_shifts[shift_idx][i]))

                atom_shifts[shift_idx][i] = sorted(atom_shifts[shift_idx][i])

            atom_shifts[shift_idx] = np.median(atom_shifts[shift_idx], 0).tolist()
        
        return atom_shifts

    

        