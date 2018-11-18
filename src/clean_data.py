'''
This script provides functions to read & analyze the quick draw data set mainly using pandas and matplotlib.
'''

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Build a data class to get the features and targets for the modeling part
class Data:

    def __init__(self, df):
        self.df = df
        self.y = self.df['target'].values
        self.X = self.df.drop(['countrycode', 'drawing', 'key_id', 'timestamp', 'word','target'], axis=1).values


def add_image(df, animal):
    df_animal_image = pd.DataFrame(animal)
    df = pd.concat([df, df_animal_image], axis = 1)
    df = df[df['recognized'] == True]
    df = df.drop(['recognized'], axis = 1)
    df = df.sample(n = 5000, replace=True)
    return df

def get_data():
    df_lion_original = pd.read_csv('../data/lion.csv')
    lion = np.load('../data_npy/lion.npy')
    df_lion = add_image(df_lion_original, lion)

    df_panda_original = pd.read_csv('../data/panda.csv')
    panda = np.load('../data_npy/panda.npy')
    df_panda = add_image(df_panda_original, panda)

    df_monkey_original = pd.read_csv('../data/monkey.csv')
    monkey = np.load('../data_npy/monkey.npy')
    df_monkey = add_image(df_monkey_original, monkey)

    df_duck_original = pd.read_csv('../data/duck.csv')
    duck = np.load('../data_npy/duck.npy')
    df_duck = add_image(df_duck_original, duck)
    return df_lion, df_panda, df_monkey, df_duck


def plot_drawing(df):
    df1 = df[1000:1040]
    example1s = [ast.literal_eval(pts) for pts in df1.drawing.values]

    fig, axes = plt.subplots(5,8, figsize=(18,6))
    axes = axes.ravel()

    for i, example in enumerate(example1s):
        for x,y in example:
            axes[i].plot(x, y, marker='.')
            axes[i].axis('off')
            if i == 40:
                break
    fig.savefig('../image/profolio_3.png', bbox_inches='tight')


def combine(df1, df2, df3, df4):
    df = pd.concat([df1, df2, df3, df4], axis = 0)
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def main():
    df_lion, df_panda, df_monkey, df_duck =get_data()
    df = combine(df_lion, df_panda, df_monkey, df_duck)
    lb_make = LabelEncoder()
    df["target"] = lb_make.fit_transform(df["word"])
    #plot_drawing(df)

if __name__ == '__main__':
    main()
