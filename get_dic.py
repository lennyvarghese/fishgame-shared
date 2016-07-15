import pymc
import hddm
import kabuki
import numpy as np
import glob
import os
import pandas


def get_dic(hddmModel):
    '''
    compute devianceBar - deviance(thetaBar)
    '''

    mcModel = hddmModel.mc
    db = mcModel.db
    # get the expected value of the deviance
    devianceBar = np.mean(db.trace('deviance')())

    # need to evaluate deviance at "best" posterior estimate
    for stochastic in mcModel.stochastics:
        try:
            traces = stochastic.trace()
            if ('trans' in stochastic.__name__):
                # because these are apparently double transformed
                bestValue = np.mean(pymc.invlogit(pymc.invlogit(traces)))
            else:
                bestValue = np.mean(traces)

            stochastic.value = bestValue

        except KeyError:
            print('No trace available for {:s}'.format(stochastic.__name__))

    dic = 2*devianceBar - mcModel.deviance 

    return dic 


def load_models(directory):
    modelList = []
    modelNames = glob.glob(os.path.join(directory, 'model*.pkl'))
    for m in modelNames:
        modelList.append(hddm.load(m))

    return kabuki.utils.concat_models(modelList)


def get_model_paths(task):
    return glob.glob(task+ '_all_accuracy_*')


if __name__ == '__main__':
    for task in ['audio', 'visual']:
        resultDir = get_model_paths(task)
        for s in ['concat', '1', '2']:
            resList = []
            for r in resultDir:
                if s == 'concat':
                    m = load_models(r)
                else:
                    m = hddm.load(os.path.join(r, 'model_{:s}.pkl'.format(s)))
                tr = m.get_traces()
                tr.to_csv(os.path.join(r, 'model_{:s}_traces.csv'.format(s)),
                          header=True, index=False)
                dic = get_dic(m)

                x = [s['outcome'] for s in m.model_descrs if 
                     'C(condition, Treatment("control")' in s['model']]
                resList.append((task, '+'.join(x), dic))

            d = pandas.DataFrame(resList, columns=['task', 'dependsOn', 'dic'])
            d = d.sort_values(by='dic')
            d.to_csv('dic_' + task + '_results_model_{:s}.csv'.format(s), 
                 header=True, index=False)
