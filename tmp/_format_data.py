"""Puts the data into a nice format and names it `data.csv`.

"""

import pandas as pd


def get_dataframe():
    """Retrieves the dataframe and formats it.

    """
    df = pd.read_csv('fishgame_data_final_with_NaN.csv').dropna()
    df.columns = [
        'subj_idx', 'first_session', 'task', 'run_num', 'condition',
        'direction_travel', 'stimulus', 'answered_direction', 'answered',
        'trial_num', 'correct', 'rt_ms'
    ]
    df['rt'] = df['rt_ms'] / 1000.
    subjects = df.subj_idx.unique().tolist()
    [df.subj_idx.replace({s: 'S%02d' % i}, inplace=True)
     for i, s in enumerate(subjects)]
    df.condition = df.condition.apply(lambda s: s.lower().rstrip())
    df.stimulus.replace({'good': 'slow', 'bad': 'fast'}, inplace=True)
    df.answered.replace({'good': 'slow', 'bad': 'fast'}, inplace=True)
    df = df[df.direction_travel == df.answered_direction]
    df = df[['subj_idx', 'task', 'condition', 'stimulus', 'answered',
             'correct', 'rt']]
    df['response'] = df.answered.replace({'slow': '0', 'fast': '1'})
    df.correct = df.correct.astype(int)

    return df


if __name__ == '__main__':

    df = get_dataframe()
    df.to_csv('data.csv', index=False)
