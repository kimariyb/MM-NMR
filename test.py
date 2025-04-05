from data import carbon

if __name__ == '__main__':
    # create dataset and filter the 100 most common molecules
    dataset = carbon.CarbonDataset('./data/dataset/carbon')
    data = dataset[0]
    print(data)
    print(data.x)
    print(data.edge_attr)
    print(data.edge_index)
    print(data.y)
    print(data.mask)
    print(data.z)
    print(data.pos)
    