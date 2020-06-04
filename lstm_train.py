import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lsmt_helper import transform_data, LSTMClassifier, Optimization


# Load features from the disk
with (open("all_features.pickle", "rb")) as openfile:
    while True:
        try:
            all_features = pickle.load(openfile)
        except EOFError:
            break


# Transform data into a format that is appropiate for modelling a LSTM

input_X, input_Y = transform_data(all_features)
input_X = np.vstack([input_X])[..., 0]
input_Y = np.vstack(input_Y)
X_train, X_valid, y_train, y_valid = train_test_split(input_X, input_Y, test_size=0.2)

# Add a pytorch wrapper to data

X_train_nrml = DataLoader(np.array(X_train), batch_size=1)
X_valid_nrml = DataLoader(np.array(X_valid), batch_size=1)

y_train_nrml = DataLoader(np.array(y_train))
y_valid_nrml = DataLoader(np.array(y_valid))

# Create model
model = LSTMClassifier(input_size=1, hidden_size=30, layer_size=3, output_size=3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

# Train model
num_epochs = 100
optimization = Optimization(model, loss_fn, optimizer, scheduler)
optimization.train(X_train_nrml, y_train_nrml, X_valid_nrml, y_valid_nrml, num_epochs)
