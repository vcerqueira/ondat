import pandas as pd

from codebase.load_data.config import DATASETS, DATA_GROUPS

summary_list = []
for ds in DATA_GROUPS:

    for group in DATA_GROUPS[ds]:
        data_loader = DATASETS[ds]

        df = data_loader.load_data(group)

        mean_size = df.groupby('unique_id').apply(lambda x: x.shape[0]).mean()
        n_ts = df['unique_id'].value_counts().__len__()
        n_vals = df.shape[0]

        horizon = data_loader.horizons_map.get(group)
        input_s = data_loader.context_length.get(group)
        freq_int = data_loader.frequency_map.get(group)

        summary_ = {
            'data': f'{ds}_{group}',
            'mean': mean_size,
            'n': n_ts,
            'n_vals': n_vals,
            # 'horizon': horizon,
            # 'input_s': input_s,
            # 'freq': freq_int,
        }

        summary_list.append(summary_)

df = pd.DataFrame(summary_list)

tot = {'data': 'total',
       'mean': '-',
       'n': df['n'].sum(),
       'n_vals': df['n_vals'].sum(), }

df = pd.concat([df, pd.DataFrame(pd.Series(tot)).T])

print(df.round(1).to_latex(caption='tab:data', label='tab:data', index=False))
