import os
import ast
import dgl
import torch
import numpy as np

from rdkit import Chem, RDLogger
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.featurizer import mol_to_geognn_data, data_to_dgl_graph


RDLogger.DisableLog('rdApp.*')


class DatasetBuilder:
    def __init__(self, root):
        self.root = root

    def build(self): 
        raise NotImplementedError("DatasetBuilder.build() is not implemented.")

    def process(self):
        raise NotImplementedError("DatasetBuilder.process() is not implemented.")

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self):
        raise NotImplementedError("DatasetBuilder.raw_file_names is not implemented.")
    
    @property
    def processed_file_names(self):
        return "processed.pt"
    
    @property
    def processed_paths(self):
        return os.path.join(self.processed_dir, self.processed_file_names)


class CarbonDatasetBuilder(DatasetBuilder):
    def __init__(self, root):
        super().__init__(root)
        
    @property
    def raw_file_names(self):
        return "carbon_dataset.sdf"

    def build(self):
        r"""
        Build the dataset.
        """
        # logging the progress
        print("Building the carbon dataset...")
        
        # check if the raw data exists
        if not os.path.exists(os.path.join(self.raw_dir, self.raw_file_names)):
            print("The raw data does not exist.")
            return None
        
        # check if the processed data exists
        if not os.path.exists(self.processed_paths):
            print("The processed data does not exist. Processing the raw data...")
            self.process()
        else:
            print("The processed data already exists. Loading the processed data...")
        
        # load the processed data
        data = torch.load(self.processed_paths)
        
        atom_bond_graphs_list = data['atom_bond_graphs_list']
        bond_angle_graphs_list =  data['bond_angle_graphs_list']
        label_list = data['label_list']
        mask_list = data['mask_list']
        
        # create the dataset
        dataset = GraphDataset(atom_bond_graphs_list, bond_angle_graphs_list, label_list, mask_list)
        
        # logging the progress
        print("The carbon dataset has been built.")
        
        return dataset
    
    def process(self):
        r"""
        Process the raw data and save it in the processed folder.
        """
        # logging the progress
        print("Processing the raw carbon data...")
        
        # Load sdf file
        suppl = Chem.SDMolSupplier(os.path.join(self.raw_dir, self.raw_file_names), removeHs=False, sanitize=True)
        print("The raw carbon data is being processed.")
        
        # initialize the lists
        atom_bond_graphs_list = []
        bond_angle_graphs_list = []
        label_list = []
        mask_list = []
        
        for mol in tqdm(suppl, desc="Processing data", total=len(suppl)):
            if mol is None:
                continue
            
            # Get graph data
            atom_bond_graph, bond_angle_graph = data_to_dgl_graph(mol_to_geognn_data(mol))
            if atom_bond_graph is None or bond_angle_graph is None:
                continue

            # get the carbon shift
            atom_shifts = self.get_carbon_shift(mol)
            if len(atom_shifts) == 0:
                continue
            
            # get the carbon atoms
            for i, atom in enumerate(mol.GetAtoms()):
                if i in atom_shifts:
                    atom.SetProp('shift', str(atom_shifts[i]))
                    atom.SetBoolProp('mask', True)
                else:
                    atom.SetProp('shift', str(0))
                    atom.SetBoolProp('mask', False)

            # get the label and mask
            shift = np.array([ast.literal_eval(atom.GetProp('shift')) for atom in mol.GetAtoms()])
            mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

            # save the data
            atom_bond_graphs_list.append(atom_bond_graph)
            bond_angle_graphs_list.append(bond_angle_graph)
            label_list.append(shift)
            mask_list.append(mask)
        
        data = {
            'atom_bond_graphs_list': atom_bond_graphs_list, 
            'bond_angle_graphs_list': bond_angle_graphs_list, 
            'label_list': label_list,
            'mask_list': mask_list
        }
        
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # save the dictionary
        torch.save(data, self.processed_paths)
        
        # logging the progress
        print("The raw carbon data has been processed and saved.")       

    def get_carbon_shift(self, mol: Chem.rdchem.Mol) -> dict:
        r"""
        Get the carbon shift of each carbon atom.
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


class GraphDataset(Dataset):
    def __init__(self, atom_bond_graphs, bond_angle_graphs, label_list, mask_list):
        # check the lengths of the lists
        assert len(atom_bond_graphs) == len(bond_angle_graphs) == len(label_list) == len(mask_list)
        # save the data
        self.atom_bond_graphs = atom_bond_graphs
        self.bond_angle_graphs = bond_angle_graphs
        self.labels = label_list
        self.masks = mask_list
        
    def __len__(self):
        return len(self.atom_bond_graphs)
    
    def __getitem__(self, idx):
        return self.atom_bond_graphs[idx], self.bond_angle_graphs[idx], self.labels[idx], self.masks[idx]
    

