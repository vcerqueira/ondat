import pandas as pd
import plotnine as p9

from codebase.experiments.load_results import ResultsReader
from codebase.experiments.analyzer import ResultAnalysis
from codebase.methods.log import LogTransformation

pd.set_option('display.max_columns', None)

cv = ResultsReader.read_results()
analysis = ResultAnalysis(cv)

perf = analysis.calc_error_by(cv, 'group')
perf = perf.iloc[:, [1, 3, 0, 2]]
# perf = analysis.calc_error_by(cv, 'freq')
# perf = analysis.calc_error_by(cv, 'unique_id')

perf.round(5)
print(perf.mean().sort_values())
print(perf.rank(axis=1).mean().sort_values())
print(perf.rank(axis=1))
# perf.boxplot()


perf.loc['Average'] = perf.mean()
perf.loc['Average Rank'] = perf.rank(axis=1).mean().round(2)

perf_str = perf.round(5).astype(str)
print(perf_str.to_latex(caption='adsad', label='tab:results'))

reference = 'Standard'
df_pd = perf.drop(columns=reference)
for c in df_pd:
    df_pd[c] = 100 * ((perf[c] - perf[reference]) / perf[reference])
    df_pd[c] = LogTransformation.transform(df_pd[c])

plot_pd = p9.ggplot(df_pd.melt(), p9.aes(x='variable', y='value')) + \
          ResultAnalysis.get_p9_theme() + \
          p9.theme(plot_margin=0.025,
                   axis_text=p9.element_text(size=12),
                   # axis_text_x=p9.element_blank(),
                   legend_title=p9.element_blank(),
                   legend_position=None) + \
          p9.geom_boxplot(fill='#66CDAA',
                          width=0.7,
                          show_legend=False) + \
          p9.labs(x='', y='Log % diff.') + \
          p9.geom_hline(yintercept=0,
                        linetype='dashed',
                        color='black',
                        size=1.1,
                        alpha=0.7)

plot_rpd = \
    p9.ggplot(df_pd.rank(axis=1).melt(), p9.aes(x='variable', y='value')) + \
    ResultAnalysis.get_p9_theme() + \
    p9.theme(plot_margin=0,
             axis_text=p9.element_text(size=12),
             # axis_text_x=p9.element_blank(),
             legend_title=p9.element_blank(),
             legend_position=None) + \
    p9.geom_boxplot(fill='#66CDAA',
                    width=0.7,
                    show_legend=False) + \
    p9.labs(x='', y='Log % diff.') + \
    p9.geom_hline(yintercept=0,
                  linetype='dashed',
                  color='black',
                  size=1.1,
                  alpha=0.7)
