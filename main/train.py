import sys
import timeit

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, r2_score

import preprocess as pp

class GraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output):
        super(GraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_output)])
        if task == 'classification':
            self.W_property = nn.Linear(dim, 2)
        if task == 'regression':
            self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, tuple(axis.cpu().numpy()))]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, tuple(axis.cpu().numpy()))]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return molecular_vectors
    
class CombinedModel(nn.Module):
    def __init__(self, gnn, homo_lumo_dim, mlp_hidden_dim, mlp_output_dim):
        super(CombinedModel, self).__init__()
        self.gnn = gnn
        self.mlp_hl = nn.Sequential(
            nn.Linear(homo_lumo_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_output_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(mlp_output_dim + gnn.W_property.in_features, 1)

    def forward(self, gnn_data, homo_lumo_data, train=True):
        gnn_inputs = gnn_data[:-1]
        
        # Check if gnn_data[-1] is a tensor
        if isinstance(gnn_data[-1], torch.Tensor):
            correct_values = gnn_data[-1].view(-1, 1)  # Adjust size to match predicted values
        else:
            correct_values = torch.cat(gnn_data[-1]).view(-1, 1)  # Adjust size to match predicted values

        molecular_vectors = self.gnn.gnn(gnn_inputs)
        homo_lumo_vectors = self.mlp_hl(homo_lumo_data)

        combined_vectors = torch.cat((molecular_vectors, homo_lumo_vectors), dim=1)
        predicted_values = self.fc(combined_vectors)

        if train:
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                predicted_values = predicted_values.to('cpu').data.numpy()
                correct_values = correct_values.to('cpu').data.numpy()
                predicted_values = np.concatenate(predicted_values)
                correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            fingerprints_batch = [fp.clone().detach().to(device) for fp in data_batch[0]]
            adjacencies_batch = [adj.clone().detach().to(device) for adj in data_batch[1]]
            molecular_sizes_batch = torch.tensor(data_batch[2], dtype=torch.long).to(device)
            homo_lumo_batch = torch.stack([hl.clone().detach().to(device) for hl in data_batch[3]]).to(device)
            correct_values_batch = torch.tensor(data_batch[4], dtype=torch.float32).to(device)
            loss = self.model((fingerprints_batch, adjacencies_batch, molecular_sizes_batch, correct_values_batch), homo_lumo_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_regressor(self, dataset):
        N = len(dataset)
        all_predicted_values = []
        all_correct_values = []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            fingerprints_batch = [fp.clone().detach().to(device) for fp in data_batch[0]]
            adjacencies_batch = [adj.clone().detach().to(device) for adj in data_batch[1]]
            molecular_sizes_batch = torch.tensor(data_batch[2], dtype=torch.long).to(device)
            homo_lumo_batch = torch.stack([hl.clone().detach().to(device) for hl in data_batch[3]]).to(device)
            correct_values_batch = torch.tensor(data_batch[4], dtype=torch.float32).to(device)
            predicted_values, correct_values = self.model((fingerprints_batch, adjacencies_batch, molecular_sizes_batch, correct_values_batch), homo_lumo_batch, train=False)
            all_predicted_values.append(predicted_values)
            all_correct_values.append(correct_values)

        all_predicted_values = np.concatenate(all_predicted_values)
        all_correct_values = np.concatenate(all_correct_values)
        r2 = r2_score(all_predicted_values, all_correct_values)
        return r2

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

if __name__ == "__main__":

    (task, dataset, radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, lr, lr_decay, decay_interval,  weight_decay,
     iteration, homo_lumo_dim, mlp_hidden_dim, mlp_output_dim, setting) = sys.argv[1:]
    (radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, decay_interval,
     iteration, homo_lumo_dim, mlp_hidden_dim,
     mlp_output_dim) = map(int, [radius, dim, layer_hidden, layer_output,
                            batch_train, batch_test,
                            decay_interval, iteration,
                            homo_lumo_dim, mlp_hidden_dim,
                            mlp_output_dim])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    (dataset_train, dataset_dev, dataset_test,
     N_fingerprints) = pp.create_datasets(task, dataset, radius, device)
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)


    print('Creating a model.')
    torch.manual_seed(1234)
    gnn = GraphNeuralNetwork(N_fingerprints, dim, layer_hidden, layer_output).to(device)
    model = CombinedModel(gnn, homo_lumo_dim, mlp_hidden_dim, mlp_output_dim).to(device)

    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = '../output/result--' + setting + '.txt'
    if task == 'regression':
        result = 'Epoch\tTime(sec)\tLoss_dev\tLoss_train\tR2_dev\tR2_train'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_dev = trainer.train(dataset_dev)
        loss_train = trainer.train(dataset_train)

        prediction_dev = tester.test_regressor(dataset_dev)
        prediction_train = tester.test_regressor(dataset_train)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about', hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_dev, loss_train, prediction_dev, prediction_train]))
        tester.save_result(result, file_result)

        print(result)

    # Test with dataset_test
    r2_test = tester.test_regressor(dataset_test)
    print(f'R^2 score on test dataset: {r2_test}')

    # Save model
    torch.save(model.state_dict(), '../model/trained_model.pth')
