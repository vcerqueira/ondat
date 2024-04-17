from neuralforecast import NeuralForecast
import pytorch_lightning as pl

from codebase.load_data.config import DATASETS, DATA_GROUPS
from codebase.methods.ondat import OnDAT_NHITS as OnDAT

from codebase.experiments.config import (TEST_SIZE,
                                         VAL_SIZE,
                                         NHITS_CONFIG)

for data_name in DATA_GROUPS:

    for group in DATA_GROUPS[data_name]:
        # if data_name != 'M4' or group != 'Monthly':
        #     continue

        print(f'Running: {data_name}, {group}')

        pl.seed_everything(123, workers=True)

        data_loader = DATASETS[data_name]
        print(data_loader.data_group)

        df = data_loader.load_data(group)
        horizon = data_loader.horizons_map.get(group)
        n_lags = data_loader.context_length.get(group)
        freq_str = data_loader.frequency_pd.get(group)
        freq_int = data_loader.frequency_map.get(group)

        models = [
            OnDAT(h=horizon,
                  input_size=n_lags,
                  period=freq_int,
                  moving_blocks=True,
                  on_train=True,
                  on_valid=True,
                  **NHITS_CONFIG),
            OnDAT(h=horizon,
                  input_size=n_lags,
                  period=freq_int,
                  moving_blocks=True,
                  on_train=True,
                  on_valid=False,
                  **NHITS_CONFIG),
            OnDAT(h=horizon,
                  input_size=n_lags,
                  period=freq_int,
                  moving_blocks=True,
                  on_train=False,
                  on_valid=True,
                  **NHITS_CONFIG),
            OnDAT(h=horizon,
                  input_size=n_lags,
                  period=freq_int,
                  moving_blocks=False,
                  on_train=True,
                  on_valid=True,
                  **NHITS_CONFIG),
            OnDAT(h=horizon,
                  input_size=n_lags,
                  period=freq_int,
                  moving_blocks=True,
                  on_train=True,
                  on_valid=True,
                  log_on_mbb=False,
                  **NHITS_CONFIG),

        ]

        nf = NeuralForecast(models=models, freq=freq_str)

        cv_df = nf.cross_validation(df=df,
                                    val_size=horizon * VAL_SIZE,
                                    test_size=horizon * TEST_SIZE,
                                    n_windows=None)

        cv_df['dataset'] = data_name
        cv_df.to_csv(f'assets/results/{data_name}_{group}_ablation.csv', index=False)
