from src.data.io import DataIO


def main():
    dataset_path = r'C:\data\TACO\data'

    data = DataIO.read_dataset(dataset_path)
    exit(0)


if __name__ == '__main__':
    main()
