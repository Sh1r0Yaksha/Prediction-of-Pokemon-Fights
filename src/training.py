import pandas as pd
import sklearn as skl
import os

parent_dir = os.path.dirname(os.getcwd())

# Reading the preprocessed data
pokemon = pd.read_csv(parent_dir + '/data/preprocessed.csv')

# Function for getting all of the features of a pokemon given their 'id'
def get_features(index_1,index_2,pokemon):
    return (pokemon.iloc[index_1-1,1:-1:1],pokemon.iloc[index_2-1,1:-1:1])