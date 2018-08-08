import numpy as np
import pandas as pd
data = pd.read_csv('shufflefile.data')
print("This is the un shuffled data", data)
df = pd.read_csv('shufflefile.data', header=0)
data2 = df.reindex(np.random.permutation(df.index))
print("\n The shuffled data is", data2)