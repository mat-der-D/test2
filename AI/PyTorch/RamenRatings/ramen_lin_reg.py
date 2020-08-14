# Ref URL:
# https://www.codexa.net/pytorch-python/


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error



ramen = pd.read_csv('ramen-ratings.csv')

mask = ramen.index[ramen['Stars'] == 'Unrated']
ramen = ramen.drop(index = mask)

ramen['Stars'] = ramen['Stars'].astype(float)

ramen = ramen.drop(columns=['Review #', 'Top Ten', 'Variety'])


Country = pd.get_dummies(ramen['Country'], prefix='Country', drop_first=True)
Brand = pd.get_dummies(ramen['Brand'], prefix='Brand', drop_first=True)
Style = pd.get_dummies(ramen['Style'], prefix='Style', drop_first=True)

ramendf = pd.concat([Country, Brand, Style], axis=1)

X = np.array(ramendf, dtype=np.float32)
y = np.array(ramen[['Stars']], dtype=np.float32)

# *****
# X.shape == (2577, 397)
# y.shape == (2577, 1)
# *****

model = nn.Linear(397, 1)

loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.6)

max_epoch = 100000
for epoch in range(max_epoch):
    # Stage1: transfer numpy array to tensor
    inputs = torch.from_numpy(X)
    targets = torch.from_numpy(y)

    # Stage2: output guessed value to calculate cost
    outputs = model(inputs)
    cost = loss(outputs, targets)

    # Stage3: back propagation
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # Stage4: print cost per 50 epochs
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, max_epoch, cost.item()))
