from data import carbon
from torch_geometric.loader import DataLoader

from models.regressor import MultiModalFusionRegressor
from models.encoders.geometry import SphereNet
from models.encoders.graph import GraphNet

if __name__ == '__main__':
    # create dataset and filter the 100 most common molecule
    dataset = carbon.CarbonDataset('./data/dataset/carbon')
    
    # create dataloader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    data = dataset[0]
    print(data.edge_attr.shape)
    
    # create model
    graph = GraphNet(
        num_layers=3,
        emb_dim=128,
    )

    # iterate over the dataset
    for data in loader:
        # forward pass
        node_out = graph(data.x, data.edge_index, data.edge_attr)
        print("graph model output shape:", node_out.shape)
    
    
    sphere = SphereNet(out_channels=32)
    
    for data in loader:
        # forward pass
        node_out = sphere(data.z, data.pos, data.batch)
        print("sphere model output shape:", node_out.shape)
        
    