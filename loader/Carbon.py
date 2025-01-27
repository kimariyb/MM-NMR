# -*- coding: utf-8 -*-
import os
import torch

from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

from utils.Features import GernerateMask, MolToGraph, MolToFingerprints, SmilesToVector
from utils.Extractor import ExtractCarbonShift

# Disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


class CarbonSpectraDataset(InMemoryDataset):
    def __init__(
        self, 
        root = None, 
        transform = None, 
        pre_transform = None, 
        pre_filter = None, 
    ):
        self.root = root
    
        super(CarbonSpectraDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'carbon_dataset.sdf'

    @property
    def processed_file_names(self):
        return 'processed.pt'

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'carbon/raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'carbon/processed')

    def process(self):
        # 读取原始数据
        suppl = Chem.SDMolSupplier(
            os.path.join(self.raw_dir, self.raw_file_names),
            removeHs=False,
            sanitize=True
        )

        data_list = []

        # 读取数据
        for mol in tqdm(suppl, desc='Processing data', unit='mol', total=len(suppl)):
            # 过滤数据
            if mol is None:
                continue
            if mol.GetNumAtoms() < 2:
                continue
            if mol.GetNumAtoms() > 100:
                continue

            # 提取碳谱数据
            carbon_shift = ExtractCarbonShift(mol)
            
            # 生成掩码和碳谱数据
            mask, shift = GernerateMask(mol, carbon_shift)

            # 提取分子图
            graph = MolToGraph(mol)
            finger_print = MolToFingerprints(mol)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            vector = SmilesToVector(smiles)

            # 创建数据对象
            data = Data()
            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.long)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.long)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.long)
            data.fingerprint = torch.tensor(finger_print, dtype=torch.float32)
            
            # add carbon shifts
            data.mask = torch.from_numpy(mask).to(torch.bool)
            data.y = torch.from_numpy(shift).to(torch.float)

            # add SMILES and vector
            data.vector = torch.from_numpy(vector).to(torch.long)
            data.smiles = smiles

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
