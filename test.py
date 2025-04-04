from data import carbon

if __name__ == '__main__':
    dataset = carbon.CarbonDataset('./data/dataset/carbon')
    data = dataset[0]
    print(data)