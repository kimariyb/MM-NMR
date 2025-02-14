# -*- coding: utf-8 -*-
import os
import ast
import torch
import numpy as np

from rdkit import Chem, RDLogger
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

from utils.Extractor import ExtractHydrogenShift
from utils.Molecular import MolToGraph, MolToFingerprints

# Disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


class HydrogenSpectraDataset(InMemoryDataset):
    def __init__(
        self, 
        root = None, 
        transform = None, 
        pre_transform = None, 
        pre_filter = None, 
    ):
        self.root = root
    
        super(HydrogenSpectraDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'hydrogen_dataset.sdf'

    @property
    def processed_file_names(self):
        return 'processed.pt'

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'hydrogen/raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root,'hydrogen/processed')
    
    def process(self) -> None:
        # 读取原始数据
        suppl = Chem.SDMolSupplier(
            os.path.join(self.raw_dir, self.raw_file_names),
            removeHs=False,
            sanitize=True
        )

        data_list = []

        # 读取数据
        for mol in tqdm(suppl, desc='Processing data', unit='mol', total=len(suppl)):
            if mol is None:
                continue
            if mol.GetNumAtoms() < 2:
                continue
            if mol.GetNumAtoms() > 100:
                continue
            
            # 提取氢谱数据
            hydrogen_shift = ExtractHydrogenShift(mol)
            
            # 标记氢原子的位置
            for i, atom in enumerate(mol.GetAtoms()):
                if i in hydrogen_shift:
                    atom.SetProp('shift', str(hydrogen_shift[i]))
                    atom.SetBoolProp('mask', True)
                else:
                    atom.SetProp('shift', str([0]))
                    atom.SetBoolProp('mask', False)

            # 提取分子图
            graph = MolToGraph(mol)

            # 创建数据对象
            data = Data()
            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.long)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.long)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.long)
            
            # add fluorine shifts
            shift = np.array([ast.literal_eval(atom.GetProp('shift')) for atom in mol.GetAtoms()])
            mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])
            data.mask = torch.from_numpy(mask).to(torch.bool)
            data.y = torch.from_numpy(shift).to(torch.float)

            data.smiles = smiles

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        
  