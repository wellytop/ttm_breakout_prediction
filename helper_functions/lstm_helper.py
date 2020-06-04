import torch
from torch.nn import functional
from sklearn.preprocessing import StandardScaler, LabelEncoder


class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, layer_size, batch_first=False
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h_t = torch.zeros(self.layer_size, x.size(1), self.hidden_size)
        c_t = torch.zeros(self.layer_size, x.size(1), self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h_t, c_t))
        out = out[-1, :, :].contiguous()
        out = self.fc(out)
        return out


class Optimization:
    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.sched = scheduler

    def _validate(self, X_valid, y_valid, loss, epoch):

        self.model.eval()
        correct, total, best_acc = 0, 0, 0
        for x_val, y_val in zip(X_valid, y_valid):
            x_val, y_val = [t for t in (x_val, y_val)]
            x_val = x_val.reshape(120, 3, -1)
            out = self.model(x_val.float())

            preds = torch.nn.functional.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0) * 3
            correct += (preds == y_val).sum().item()

        acc = correct / total

        if epoch % 5 == 0:
            print(f"Epoch: {epoch}. Loss: {loss.item():.4f}. Acc: {acc:2.2%}")

        if acc > best_acc:
            trials = 0
            best_acc = acc
            print(f"Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}")

    def train(self, X_train, y_train, X_valid, y_valid, num_epochs):
        for epoch in range(1, num_epochs + 1):
            for i, (x_batch, y_batch) in enumerate(zip(X_train, y_train)):
                x_batch = x_batch.reshape(120, 3, -1)
                y_batch = y_batch[0]
                self.model.train()
                out = self.model(x_batch.float())
                self.opt.zero_grad()
                loss = self.loss_fn(out, y_batch)
                loss.backward()
                self.opt.step()

            self.sched.step()

            self._validate(X_valid, y_valid, loss, epoch)

    # def _evaluate():


def transform_data(data):

    input_X = []
    input_Y = []

    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    label_encoder.fit(["SELL", "BUY"])

    for ticker in data.keys():
        print(ticker)
        time_series_x = []
        label_y = []
        batch_size = 0
        for feature in data[ticker].keys():
            stock_df = data[ticker][feature]["data"]
            label = data[ticker][feature]["label"]

            ind = stock_df.index.get_loc(0)
            stock_target_df = stock_df.iloc[:ind]

            stock_target_df_ttm = stock_df.iloc[ind:]

            time_series = stock_target_df["close"].values
            if label != "ambiguous" and len(time_series) == 120:
                print(feature)
                x = scaler.fit_transform(time_series.reshape(-1, 1))
                time_series_x.append(x)
                label = label_encoder.transform([label])
                label_y.append(label[0])

                batch_size += 1
                if batch_size == 3:
                    break

        if len(time_series_x) == 3:
            input_X.append(time_series_x)
            input_Y.append(label_y)

    return input_X, input_Y
