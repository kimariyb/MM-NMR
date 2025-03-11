from utils.loader import CarbonDatasetBuilder

if __name__ == '__main__':
    dataset_builder = CarbonDatasetBuilder('data/carbon')
    dataset = dataset_builder.build()
    print(len(dataset))