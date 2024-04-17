import pandas as pd
import plotnine as p9

from codebase.experiments.load_results import ResultsReader, LogReader
from codebase.experiments.analyzer import ResultAnalysis

pd.set_option('display.max_columns', None)

# testing performance

cv = ResultsReader.read_results()
analysis = ResultAnalysis(cv)

test = analysis.calc_error_by(cv, 'group')

# validation performance estimation

valid = LogReader.read_results()

perf_est = []
for idx, r in valid.dropna().iterrows():
    print(idx)

    # pd_v2t = 100 * ((r - test.loc[idx]) / test.loc[idx])
    pd_v2t = r - test.loc[idx]

    perf_est.append(pd_v2t)

pe_df = pd.concat(perf_est, axis=1).T.dropna(axis=1)

df = pe_df.median()
df = df.reset_index()
df.columns = ['Method', 'SMAPE diff.']
df['Method'] = pd.Categorical(df['Method'], categories=['NH_OnDAT_A3',
                                                        'NHITS',
                                                        'DA+Standard'])

plot_pd = p9.ggplot(df, p9.aes(x='Method', y='SMAPE diff.')) + \
          ResultAnalysis.get_p9_theme() + \
          p9.theme(plot_margin=0.025,
                   axis_text=p9.element_text(size=12),
                   # axis_text_x=p9.element_blank(),
                   legend_title=p9.element_blank(),
                   legend_position=None) + \
          p9.geom_bar(stat='identity', fill='steelblue') + \
          p9.labs(x='', y='SMAPE diff.')

plot_pd.save('assets/perf_estim.pdf', width=10, height=5)

pe_df.loc['Median', :] = pe_df.median()

print(pe_df.round(3).astype(str).to_latex(caption='adsa',label='tab:pe'))
