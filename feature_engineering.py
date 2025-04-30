import pandas as pd

def apply_feature_engineering(df):
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    df.drop(['u','g','r','i','z'],axis=1,inplace=True)
    

    return df