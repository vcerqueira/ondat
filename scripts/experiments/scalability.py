# measure time it takes to complete 100 training steps
import time

import pandas as pd
from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast
from lightning.pytorch.loggers import CSVLogger
import pytorch_lightning as pl

from codebase.load_data.config import DATASETS, DATA_GROUPS
from codebase.methods.ondat import OnDAT_NHITS as OnDAT
from codebase.methods.mbb import TimeSeriesBootstrap

from codebase.experiments.config import (TEST_SIZE,
                                         VAL_SIZE,
                                         MOVING_BLOCKS,
                                         LOG,
                                         NHITS_CONFIG)

NHITS_CONFIG['max_steps'] = 100

execution_times_d = {}
for data_name in DATA_GROUPS:
    # if data_name != 'Gluonts':
    #     continue

    for group in DATA_GROUPS[data_name]:
        # if group != 'Monthly':
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

        # Bootstrapping APRIORI

        execution_times = {}

        start = time.time()

        boot = TimeSeriesBootstrap(log=LOG,
                                   period=freq_int,
                                   moving_blocks=MOVING_BLOCKS)

        augmented_df = boot.apriori_augment_df(df, horizon * TEST_SIZE)

        models_da = [NHITS(h=horizon,
                           input_size=n_lags,
                           logger=CSVLogger(save_dir="assets/logs",
                                            name=f'DA_{data_name}_{group}'),
                           **NHITS_CONFIG)]

        nf = NeuralForecast(models=models_da, freq=freq_str)

        cv_nf_preda = nf.cross_validation(df=augmented_df,
                                          val_size=horizon * VAL_SIZE,
                                          test_size=horizon * TEST_SIZE,
                                          n_windows=None)

        end = time.time()
        execution_times['DA+Standard'] = end - start

        # nhits

        start = time.time()

        models = [
            NHITS(h=horizon, input_size=n_lags, **NHITS_CONFIG),
        ]

        nf = NeuralForecast(models=models, freq=freq_str)

        cv_nf = nf.cross_validation(df=df,
                                    val_size=horizon * VAL_SIZE,
                                    test_size=horizon * TEST_SIZE,
                                    n_windows=None)

        end = time.time()
        execution_times['Standard'] = end - start

        # ondat

        start = time.time()

        models = [
            OnDAT(h=horizon,
                  input_size=n_lags,
                  period=freq_int,
                  moving_blocks=True,
                  on_train=True,
                  on_valid=True,
                  **NHITS_CONFIG),

        ]

        nf = NeuralForecast(models=models, freq=freq_str)

        cv_nf = nf.cross_validation(df=df,
                                    val_size=horizon * VAL_SIZE,
                                    test_size=horizon * TEST_SIZE,
                                    n_windows=None)

        end = time.time()
        execution_times['OnDAT'] = end - start

        execution_times_d[f'{data_name}_{group}'] = execution_times

exec_df = pd.DataFrame(execution_times_d).T
exec_df.to_csv('assets/results/execution_time.csv')

exec_dt_df = exec_df.copy()
for col in exec_df:
    exec_dt_df[col] = 100 * ((exec_dt_df[col] - exec_dt_df['OnDAT']) / exec_dt_df['OnDAT'])

med_exec = exec_dt_df.median()
med_exec = pd.DataFrame(med_exec).T

print(med_exec.to_latex(caption='adsa', label='tab:exec'))
