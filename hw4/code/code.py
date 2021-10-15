# This is the file you should use to write your code for the implementation problem.
# Do not change the name of this file.
# Please do not include print statements outside the provided functions, as this may crash the autograder.
# You may add any function definition as you see appropriate.
# The file should not contain the if __name__ == '__main__' block upon submission (You can comment it out if you used it in your development process).

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add imports below as you see fit. The packages you are allowed to use include:
# Pytorch, scikit-learn, NumPy, matplotlib


# This is the provided Dataloader class. You SHOULD NOT change this.
# Tip: You can load your training data with the following lines:
# train_data = MyData('train')
# train_loader = DataLoader(train_data,batch_size=BATCH_SIZE)
# where you define the constant BATCH_SIZE beforehand
# For more information on the Dataloader, see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class MyData(Dataset):
    def __init__(self, mode):
        with open(mode+'.pkl', 'rb') as handle:
            data = pickle.load(handle)
            self.X = data['x'].astype('float')
            self.y = data['y'].astype('long')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.X[idx], self.y[idx])

        return sample

# Do NOT change the headline of the class.
# This class should contain your model architecture.
# Usually this means it should contain an init function and a forward function.
# Note that things like training and evaluation
# should be put under a separate function definition OUTSIDE this class.
# Please make sure we can load your model with:
# model = MyModel()
# This means you must give default values to all parameters you may wish to set, such as output size.
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d1 = nn.Conv1d(1, 20, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv1d2 = nn.Conv1d(20, 10, kernel_size=1, padding=0)
        self.maxpool2 = nn.MaxPool1d(3, stride=1, padding=1)
        self.fc1 = nn.Linear(400, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.type(torch.float32)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        x = torch.relu(self.conv1d1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv1d2(x))
        x = self.maxpool2(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        x = torch.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


# You SHOULD NOT change this function. Instead, use it to save your TRAINED model properly.
# The parameter [model] is the TRAINED torch model you wish to save.
# For more information, see https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_model(model):
    torch.save(model.state_dict(), "mymodel.pth")


def train_one_epoch(train_loader, optimizer, model, criterion):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # print(loss)

        loss.backward()
        optimizer.step()


def test(test_loader, model):
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, prediction = torch.max(outputs, 1)
        total += labels.shape[0]
        correct += torch.sum(torch.eq(prediction, labels)).item()

    # print(correct / total)


# # We are going to load your model with:
# def main():
#     train_data = MyData('train')
#     train_loader = DataLoader(train_data, batch_size=64)
#     test_loader = DataLoader(train_data, batch_size=64)
#
#     model = MyModel()
#     model.train()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#
#     num_epoch = 100
#     for i in range(num_epoch):
#         train_one_epoch(train_loader, optimizer, model, criterion)
#     save_model(model)
#
#     model.load_state_dict(torch.load("mymodel.pth"))
#     model.eval()
#     test(test_loader, model)

#
# if __name__ == '__main__':
#     main()

# # And run your model on our test set, then grade based on your test accuracy
