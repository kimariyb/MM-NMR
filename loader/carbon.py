import os
import re
import torch
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected, coalesce
from torch_geometric.nn import radius_graph
from loader.utils import MoleculeFilter, MoleculeEncoder, generate_3d_conformation
from typing import Dict, Optional


class CarbonNMRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super(CarbonNMRDataset, self).__init__(root, transform, pre_transform, pre_filter)
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
        # 读取原始数据成 
        suppl = Chem.SDMolSupplier(
            os.path.join(self.raw_dir, self.raw_file_names), removeHs=False
        )

        # 初始化分子过滤器
        molecule_filter = MoleculeFilter(
            forbidden_fragments=[],
            min_heavy_atoms=1,
            max_heavy_atoms=45,
            min_mol_weight=10.0,
            max_mol_weight=300.0
        )

        # 初始化分子编码器
        molecule_encoder = MoleculeEncoder()

        for mol in tqdm(
            suppl, desc="Generating carbon dataset", 
            total=len(suppl), 
        ):
            # 过滤掉不满足条件的分子
            if mol is None:
                continue
            if not molecule_filter.is_valid_molecule(mol):
                continue
                
            # 获取分子信息
            mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
            mol_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

            # 获取 NMR Spec
            nmr_spec = self.getSpec13C(mol)
            # 如果没有 NMR Spec 就直接跳过
            if nmr_spec is None:
                continue
            
            nmr_list = []

            # 初始化 atom_feats 和 bond_feats
            atom_feats = []
            bond_feats = []

            # 获取原子特征
            for atom in mol.GetAtoms():
                # 编码原子特征
                atom_feats.append(molecule_encoder.encode_atom(atom))
                # 获取原子的 NMR Spec
                if atom.GetIdx() in nmr_spec:
                    nmr_list.append(nmr_spec[atom.GetIdx()])
                else:
                    nmr_list.append(-1)

            nmr_list = torch.tensor(nmr_list, dtype=torch.float) # [N,]，N 为原子个数
            atom_feats = torch.tensor(atom_feats, dtype=torch.float) # [N, 24]，N 为原子个数

            # 获取边特征
            row, col = [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start]
                col += [end]
                bond_feats.append(molecule_encoder.encode_bond(bond))

            bond_feats = torch.tensor(bond_feats, dtype=torch.float) # [M, 8]，M 为边个数
            edge_index_b = torch.tensor([row, col], dtype=torch.long)
            edge_index_b, bond_feats = to_undirected(edge_index_b, bond_feats)

            # 生成 3D 构象
            pos, _, _ = generate_3d_conformation(mol)
            pos = torch.from_numpy(pos).dtype(torch.float)
            edge_index_r = radius_graph(pos, 1000.0)
            edge_index_r = to_undirected(edge_index_r)
            
            edge_index = torch.column_stack([edge_index_b, edge_index_r])
            edge_attr = torch.cat([
                bond_feats, torch.zeros(edge_index_r.size(1))
            ])

            edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='max')

            data = Data()
            data.x = atom_feats
            data.y = nmr_list
            data.edge_attr = edge_attr
            data.edge_index = edge_index
            data.pos = pos
            data.name = mol_name
            data.smiles = mol_smiles

            if self.pre_filter is not None:
                data = self.pre_filter(data)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
           

    def getSpec13C(self, mol: Chem.Mol) -> Optional[Dict[int, float]]:
        r"""
        获取 Mol 对象中包含的 13C NMR Spectru

        Parameters
        ----------
        mol : Chem.Mol
            一个 RDKit Mol 对象

        Returns
        -------
        spec : dict
            一个字典，键为原子索引，值为对应的 NMR Spec
        """
        # 首先拿到 mol 对象中所有的 Prop
        props = mol.GetPropsAsDict()

        # 初始化一个空字典
        spectra = {}

        spectrum_keys = [key for key in props.keys() if key.startswith("Spectrum 13C")]
        
        if not spectrum_keys:
            return None

        for key in spectrum_keys:
            # 获取键对应的值
            spectrum_match = re.match(r"Spectrum 13C (\d+)", key)
            if not spectrum_match:
                return None
            
            spectrum_idx = int(spectrum_match.group(1))

            # 获取键对应的值
            spectrum_str  = mol.GetProp(key)

            # 解析 NMR 数据
            atom_shifts = self.parseSpec13C(spectrum_str)

            # 将数据添加到字典中
            for atom_idx, shift in atom_shifts.item():
                if atom_idx not in spectra:
                    spectra[atom_idx] = []
                spectra[atom_idx].append(shift)
        
        # 计算每个原子的平均化学位移
        avg_shifts = {}
        for atom_idx, shifts in spectra.items():
            avg_shifts[atom_idx] = np.mean(shifts)

        return avg_shifts
    
    def parseSpec13C(spectrum_line: str) -> Dict:
        atom_shifts = {}
        
        # 分割每个数据点
        for shift_data in spectrum_line.split('|'):
            if not shift_data:
                continue
                
            # 每个数据点的格式是 "shift_value;intensity;atom_index"
            parts = shift_data.split(';')
            try:
                shift_value = float(parts[0])
                atom_index = int(parts[-1])
                atom_shifts[atom_index] = shift_value
            except (ValueError, IndexError):
                continue
        
        return atom_shifts