import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.series import Series

df = pd.read_csv('logs/large-scale.csv', header='infer')

df.to_latex('large-scale.tex')


n_values, K_values = list(df['n'].unique()), list(df['K'].unique())

# for each (n,K), find its running time mean and se
res = df.groupby(['n', 'K'], as_index=True).agg({'total_time': ['mean', 'std']})
res.columns = res.columns.droplevel(0)
res['std'] = res['std'] / np.sqrt(len(n_values))
res.rename(columns={'std': 'se'}, inplace=True)

# now construct the output 
columns = [r'$n \ K$'] + K_values
data = {r'$n \ K$': n_values}
for K in K_values:
    data[K] = [ r'${:.3f} \pm {:.3f}$'.format(*tuple(res.loc[(n,K)])) for n in n_values]

output_df = pd.DataFrame(data, columns = columns)
output_df.set_index(r'$n \ K$', inplace=True)
output_df.to_latex('large-scale.tex', )

with open('large-scale.tex', 'w') as ff:
    # no lines in the middle; less sig. digits
    ff.write(r'\begin{tabular}{|c||*{%s}{c|}}\hline' % len(K_values)), ff.write('\n')
    ff.write(r'\backslashbox{$n$}{$K$}'), ff.write('\n')
    ff.write('& ' + ' & '.join(['$' + str(xx) + '$' for xx in K_values]) + r'\\ \hline \hline'), ff.write('\n')
    for n in n_values:
        long_string = str(n) + ' & ' + ' & '.join(['${:.2f} \pm {:.2f}$'.format(*tuple(res.loc[(n,K)])) for K in K_values]) + r'\\ \hline'
        ff.write(long_string), ff.write('\n')
    ff.write(r'\end{tabular}')