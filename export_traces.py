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
    incong = 'C(condition, Treatment("control"))[T.incongruent]' 
    cong = 'C(condition, Treatment("control"))[T.congruent]'
    for task in ['audio', 'visual']:
        resultDir = get_model_paths(task)
        resList = []
        for r in resultDir:
            m = load_models(r)
            tr = m.get_group_traces()
            tr.rename(columns=lambda x: x.replace(incong, '_incongruent'),
                      inplace=True)
            tr.rename(columns=lambda x: x.replace(cong, '_congruent'),
                      inplace=True) 
            tr.to_csv(os.path.join(r, 'concatenated_traces.csv'),
                      header=True, index=False)
