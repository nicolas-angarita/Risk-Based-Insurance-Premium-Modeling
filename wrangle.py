import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def insurance(data):
    
    df = pd.read_csv(data)

    numerical = df.select_dtypes(include='number').columns.tolist()
    categorical = df.select_dtypes(exclude='number').columns.tolist()
    to_encode = ('sex', 'smoker')
    
    categorical = [col for col in categorical if col not in to_encode]

    for cat in to_encode:
        df[f'{cat}_encoded'] = df[cat].apply(lambda x: 0 if x in ['no', 'female'] else 1)
    
    dummies_df = pd.get_dummies(df['region'], drop_first=False, dtype='int')

    df = pd.concat([df,dummies_df], axis = 1)


    return df





def data_split(df):

    seed = 1722
    
    train, test= train_test_split(df, test_size= .3, random_state = seed)
    validate, test = train_test_split(test, test_size = 0.5, random_state = seed)

    return train, validate, test
