import pandas as pd
from neuralforecast.models import NHITS
from neuralforecast.losses.numpy import smape
from neuralforecast import NeuralForecast
from lightning.pytorch.loggers import CSVLogger
import pytorch_lightning as pl
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast

from codebase.load_data.config import DATASETS, DATA_GROUPS
from codebase.methods.ondat import OnDAT_NHITS as OnDAT
from codebase.methods.mbb import TimeSeriesBootstrap

from codebase.experiments.config import (TEST_SIZE,
                                         VAL_SIZE,
                                         MOVING_BLOCKS,
                                         LOG,
                                         NHITS_CONFIG)

for data_name in DATA_GROUPS:
    # if data_name != 'M4':
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

        boot = TimeSeriesBootstrap(log=LOG,
                                   period=freq_int,
                                   moving_blocks=MOVING_BLOCKS)

        augmented_df = boot.apriori_augment_df(df, horizon * TEST_SIZE)

        # Classical approaches

        stats_models = [SeasonalNaive(season_length=freq_int)]
        sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)

        cv_sf = sf.cross_validation(df=df,
                                    h=horizon,
                                    test_size=horizon * TEST_SIZE,
                                    n_windows=None)

        # neural

        models = [
            NHITS(h=horizon, input_size=n_lags, **NHITS_CONFIG,
                  logger=CSVLogger(save_dir="assets/logs",
                                   name=f'NHITS_{data_name}_{group}')),
            OnDAT(h=horizon,
                  input_size=n_lags,
                  period=freq_int,
                  moving_blocks=True,
                  on_train=True,
                  on_valid=True,
                  logger=CSVLogger(save_dir="assets/logs",
                                   name=f'OnDAT_{data_name}_{group}'),
                  **NHITS_CONFIG),

        ]

        nf = NeuralForecast(models=models, freq=freq_str)

        cv_nf = nf.cross_validation(df=df,
                                    val_size=horizon * VAL_SIZE,
                                    test_size=horizon * TEST_SIZE,
                                    n_windows=None)

        cv_df = cv_nf.merge(cv_sf.drop(columns=['y']), how='left', on=['unique_id', 'ds', 'cutoff'])
        cv_df = cv_df.rename(columns={'NHITS': 'Standard'})

        # DA before training -- needs a different neuralforecast instance

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

        cv_nf_preda = cv_nf_preda.rename(columns={'NHITS': 'DA+Standard'})

        cv_df = cv_df.merge(cv_nf_preda.drop(columns=['y']), how='left', on=['unique_id', 'ds', 'cutoff'])

        cv_df['dataset'] = data_name
        cv_df.to_csv(f'assets/results/{data_name}_{group}_results.csv', index=False)

        print(cv_df.columns)
        err = {
            'Standard': smape(cv_df['y'], cv_df['Standard']),
            'OnDAT': smape(cv_df['y'], cv_df['OnDAT']),
            'DA+Standard': smape(cv_df['y'], cv_df['DA+Standard']),
            'SeasonalNaive': smape(cv_df['y'], cv_df['SeasonalNaive']),
        }

        print(pd.Series(err).sort_values())
