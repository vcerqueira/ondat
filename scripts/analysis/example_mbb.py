import pandas as pd
from numerize import numerize
import plotnine as p9
import matplotlib

matplotlib.use('TkAgg')

from codebase.load_data.config import DATASETS
from codebase.methods.mbb import TimeSeriesBootstrap

group = 'Monthly'
UID = 'M1'
data_loader = DATASETS['M3']

df = data_loader.load_data(group)
freq_int = data_loader.frequency_map.get(group)

boot = TimeSeriesBootstrap(log=True,
                           period=freq_int,
                           moving_blocks=True)

df = df.query('unique_id=="M1"')
df['unique_id'] = 'Original'

synthetic_df = boot.bootstrap_df(df)
synthetic_df['unique_id'] = 'Synthetic 1'

synthetic_df2 = boot.bootstrap_df(df)
synthetic_df2['unique_id'] = 'Synthetic 2'

a_df = pd.concat([df, synthetic_df, synthetic_df2]).reset_index(drop=True)

labs = lambda lst: [numerize.numerize(x) for x in lst]

plot = \
    p9.ggplot(a_df) + \
    p9.aes(x='ds', y='y') + \
    p9.facet_grid('unique_id ~.', scales='free') + \
    p9.theme_538(base_family='Palatino', base_size=12) + \
    p9.theme(plot_margin=.025,
             axis_text_y=p9.element_text(size=10),
             panel_background=p9.element_rect(fill='white'),
             plot_background=p9.element_rect(fill='white'),
             strip_background=p9.element_rect(fill='white'),
             legend_background=p9.element_rect(fill='white'),
             strip_text=p9.element_text(size=13),
             axis_text_x=p9.element_text(size=10)) + \
    p9.geom_line(color='steelblue', size=1) + \
    p9.labs(x='', y='value') + \
    p9.scale_y_continuous(labels=labs)

print(plot)

plot.save('assets/plot_example.pdf', width=10, height=7.5)
