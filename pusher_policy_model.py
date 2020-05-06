import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils import data

import numpy as np

class PusherPolicyNet(torch.nn.Module):
    def __init__(self):
        super(PusherPolicyNet, self).__init__()
        self.fc1 = nn.Linear(9, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PusherPolicyModel:
    def __init__(self):
        self.net = PusherPolicyNet()

        train_dir = './expert.npz'
        bsize = 50

        dataset = np.load('./expert.npz')
        tensor_dataset = data.TensorDataset(torch.Tensor(dataset['obs']), torch.Tensor(dataset['action']))
        self.train_loader = data.DataLoader(tensor_dataset, batch_size=bsize, shuffle=True)


    def train(self, num_epochs=2):
        criterion = nn.MSELoss()
        #optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        optimizer=optim.Adadelta(self.net.parameters())

        valid_losses = np.zeros(num_epochs+1)
        train_losses = np.zeros(num_epochs+1)

        train_loss = self.eval(criterion)
        valid_loss = 0.0
        print('epoch 0: train loss ', train_loss, ', validation loss ', valid_loss)
        train_losses[0] = train_loss
        valid_losses[0] = valid_loss

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                obs = data[0]
                action = data[1]
                inputs = obs

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs.float())
                loss = criterion(outputs.double(), action.double())
                loss.backward()
                optimizer.step()

                # print statistics
                #running_loss += loss.item()
                #if i % 10 == 9:    # print every 2000 mini-batches
                #    print('[%d, %5d] loss: %.6f' %
                #          (epoch + 1, i + 1, running_loss / 2000))
                #    running_loss = 0.0

            # evaluate loss
            train_loss = self.eval(criterion)
            valid_loss = 0.0
            print('epoch ', epoch+1, ': train loss ', train_loss, ', validation loss ', valid_loss)
            train_losses[epoch+1] = train_loss
            valid_losses[epoch+1] = valid_loss

        print('Finished Training')
        return train_losses, valid_losses

    def eval(self, criterion):
        self.net.eval()
        train_loss = 0
        valid_loss = 0

        for data in self.train_loader:
            obs = data[0]
            action = data[1]
            inputs = obs
            output = self.net(inputs.float())
            loss = criterion(output,action)
            train_loss += loss.item()

        train_loss = train_loss/len(self.train_loader.dataset)

        return train_loss

    def infer(self, obs):
        with torch.no_grad():
            x = torch.from_numpy(obs).float()
            return self.net(x)

    def save(self, PATH):
        torch.save(self.net.state_dict(), PATH)

    def load(self, PATH):
        self.net.load_state_dict(torch.load(PATH))
        self.net.eval()


