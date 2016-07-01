"""Contains functions that return models.

"""
import numpy as np
import pandas as pd
from patsy import dmatrix as dm
import hddm


def model_stimcoding(dep_on_cong):
    """Create a model with stimulus coding.

    """
    df = pd.read_csv('data.csv', index_col=0)
    des = ['1', '1 + C(condition, Treatment("control"))']

    if 'z' not in dep_on_cong:

        params = 'atv'

    else:

        params = 'atvz'

    def flipv(x, df=df):
        """Link function used to flip v. Lower boundary (-1) becomes slow
        and upper
        boundary (1) becomes fast.

        """
        stim = (np.asarray(dm(
            '0 + C(s, [[-1], [1]])', {'s': df.stimulus.ix[x.index]}
        )))

        return x * stim

    def invlogitz(z):

        return 1 / (1 + np.exp(-z))

    lfs = {
        'v': flipv, 'z': invlogitz, 'a': lambda a: a, 't': lambda t: t
    }
    models = [
        {'model': '%s ~ %s' % (p, des[p in dep_on_cong]), 'link_func': lfs[p]}
        for p in params
    ]
    m = hddm.HDDMRegressor(
        data=df,
        models=models,
        include=['z', 'p_outlier', 'st'],
        group_only_regressors=False,
        keep_regressor_trace=True,
        group_only_nodes=['p_outlier', 'st'],
        std_depends=True,
        informative=False
    )

    return m


def run():


    m = model_stimcoding(['v'])
    m.sample(10000, burn=5000, dbname=path, db='txt')
    m.gen_stats(fname='%s/stats.csv' % path, print_hidden=True)
    ppc = hddm.utils.post_pred_gen(m, samples=100)
    ppc.to_csv('%s/ppc.csv' % path)

if __name__ == '__main__':
