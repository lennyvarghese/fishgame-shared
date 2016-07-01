import os
import sys
import pandas
import hddm
from patsy import dmatrix
import numpy as np
import inspect

##########################################
# job id number
jobNum = sys.argv[1]
# accuracy coding or stim coding
responseType = sys.argv[2] 
if jobNum != 'get_df':
    # audio or visual task
    task = str(sys.argv[3]).lower()
    # total number of samples run
    nSamples = int(sys.argv[4])
    # mcmc samples to burn
    nBurn = int(sys.argv[5])
    # "all" to include sz, sv, st; "quick" otherwise (and t --> group_only)
    includes=sys.argv[6]

    if includes == 'all':
        inc=('z', 'sz', 'sv', 'st')
        groupOnly=('sz', 'sv', 'st')
    elif includes == 'quick':
        inc=('z')
        groupOnly = ()

    # parameters that depend on congruence
    try:
        params = sys.argv[7:] 
    except IndexError:
        raise IndexError('must specify at least one parameter ' +
                         '[v(SC), a(SC), z(SC), t(SC)]')

    if len(params) != 4 or len(set(params)) != len(params):
        raise ValueError('must specify four unique parameters ' +
                         '[v(SC), a(SC), z(SC), t(SC)]')
else:
    task = None
    nSamples = None
    nBurn = None
    includes = None
    params = None
####################################


def get_data_frame():
    df = pandas.read_csv('fishgame_data_final_with_NaN.csv')
    print('Original size: {:d}'.format(df.shape[0]))
    df = df.dropna() 
    print('After dropna(): {:d}'.format(df.shape[0]))
    df.columns = ['subj_idx', 'first_session', 'task', 'run_num', 'condition',
                   'direction_travel', 'stimulus', 'answered_direction',
                   'answered', 'trial_num', 'correct', 'rt_ms']
    df['rt'] = df['rt_ms'] / 1000.
    
    sID = 1
    for s in df.subj_idx.unique():
        df.subj_idx.replace({s: 's{:02d}'.format(sID)}, inplace=True)
        sID += 1

    df.condition.replace(
                {c: c.lower().rstrip() for c in df.condition.unique()},
                 inplace=True)
    df.stimulus.replace({'good': 'slow', 'bad': 'fast'}, inplace=True)
    df.answered.replace({'good': 'slow', 'bad': 'fast'}, inplace=True)

    goodIdx = (df.direction_travel == df.answered_direction)
    print('Final size: {:d}'.format(df[goodIdx].shape[0]))
    print('\n\n\nBad rows: {:d}'.format(df[~goodIdx].shape[0]))
    print(df[~goodIdx])
    
    df = df[goodIdx]
    
    df = df.drop('first_session', axis=1)
    df = df.drop('run_num', axis=1)
    df = df.drop('trial_num', axis=1)
    df = df.drop('answered_direction', axis=1)
    df = df.drop('direction_travel', axis=1)
    df = df.drop('rt_ms', axis=1)

    return df 



# get the data frame
df = get_data_frame()
# rename the data columns based on whether stim coding or response coding is
# desired
if responseType == 'response':
    df.columns = ['subj_idx', 'task', 'condition', 'stimulus', 
                  'response', 'correct', 'rt']
elif responseType == 'accuracy':
    df.columns = ['subj_idx', 'task', 'condition', 'stimulus', 
                   'subject_answered', 'response', 'rt']


def stim_code_z(x, data=df):
    stim = (np.asarray(dmatrix('0 + C(s, [[1], [-1]])',
                               {'s': data.stimulus.ix[x.index]})))

    return 1 / (1 + np.exp(-(x * stim)))


def stim_code_v(x, data=df):
    stim = (np.asarray(dmatrix('0 + C(s, [[1], [-1]])',
                               {'s': data.stimulus.ix[x.index]})))

    return x * stim


def return_model_params(var, congruence=True, stimCode=False):

    if var not in ['v', 'z', 'a', 't']:
        raise ValueError('var must be one of ' + 
                         '"v", "z", "a", "t"')
    
    if congruence:
        trtStr = '1 + C(condition, Treatment("control"))'
    else:
        trtStr = '1'

    if var == 'v' and stimCode:
        linkFunc = stim_code_v
        lftxt = 'stim_code_v'
    elif var == 'z' and stimCode:
        linkFunc = stim_code_z 
        lftxt = 'stim_code_z'
    elif var == 'z':
        linkFunc = lambda z: 1/ (1 + np.exp(-z))
        lftxt = 'lambda z: 1/ (1 + np.exp(-z))'
    else:
        lftxt = 'lambda {:s}: {:s}'.format(var, var)
        linkFunc = eval(lftxt)

    print('Adding {:s} with link function:\n{:s}'.format(var, lftxt))

    reg = {'model': '{:s} ~ {:s}'.format(var, trtStr),
           'link_func': linkFunc}

    return reg


def run_model(i, task, params, nSamples, nBurn):
    
    print('---', i, task, params, '---')
        
    savePath = (task + '_' + includes + '_' + responseType + '_' +
                '_'.join(p for p in params))
    samplesFilename = os.path.join(savePath,
                                  'samples_{:s}.pkl'.format(i))
    modelFilename = os.path.join(savePath,
                                 'model_{:s}.pkl'.format(i))
    
    try:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
    except OSError as e:
        if e.errno != 17:
            raise OSError(e)

    reg = []
    for p in params:
        if 'C' in p:
            congruence = True
        else:
            congruence = False

        if 'S' in p:
            stimCode = True
        else:
            stimCode = False
        
        pFix = p.replace('S', '').replace('C', '')
        reg.append(return_model_params(pFix,
                                       congruence,
                                       stimCode))

    if not os.path.exists(modelFilename):
        model = hddm.HDDMRegressor(df[df.task==task],
                                   reg,
                                   include=inc + ['p_outlier', 'z'],
                                   group_only_regressors=False,
                                   group_only_nodes=groupOnly + ['p_outlier'],
                                   std_depends=True,
                                   keep_regressor_trace=True,
                                   informative=False)
        model.find_starting_values()
        print('{:d} iterations, will burn {:d}'.format(nSamples, nBurn))
        model.sample(nSamples, 
                     burn=nBurn,
                     dbname=samplesFilename,
                     db='pickle')
        model.save(modelFilename)

    return


if __name__ == "__main__" and jobNum == 'get_df':
    with open('{:s}_data.csv'.format(responseType), 'w') as s:
        df.to_csv(s, header=True, index=False)
elif __name__ == "__main__":
    run_model(jobNum, task, params, nSamples, nBurn)
