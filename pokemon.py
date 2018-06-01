import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('pokemon.csv')

lvlmod = ((2 * 100) + 10)/250
investida = 40

df['danox1'] = ((df['Attack'] / df['Defense']) * investida * lvlmod) +2


plt.scatter(df['#'], df['danox1'])
plt.show()