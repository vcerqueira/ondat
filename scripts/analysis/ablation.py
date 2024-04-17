import pandas as pd

from codebase.experiments.load_results import AblationResultsReader
from codebase.experiments.analyzer import ResultAnalysis

pd.set_option('display.max_columns', None)

cv = AblationResultsReader.read_results()
analysis = ResultAnalysis(cv)

perf = analysis.calc_error_by(cv, 'group')

perf.loc['Average'] = perf.mean()
perf.loc['Average Rank'] = perf.rank(axis=1).mean().round(2)

print(perf)
print(perf.head(-2).mean().sort_values())

perf_str = perf.round(4).astype(str)

print(perf_str.iloc[:,:-1].to_latex(caption='adsad', label='tab:ablation'))
