class LoadDataset:
    DATASET_PATH = 'assets/datasets'
    DATASET_NAME = ''

    horizons = []
    frequency = []
    horizons_map = {}
    frequency_map = {}
    context_length = {}
    frequency_pd = {}
    data_group = [*horizons_map]

    @classmethod
    def load_data(cls, group):
        pass
