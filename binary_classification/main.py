"""
    Simple binary classification system
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

def plot_dataset(title, labels, dataset_x, dataset_y=None):
    """
        plots dataset
        can be a dataset in one or two dimensions
        title : title of the plot
        labels : list of dataset labels organized the same way of dataset_x and dataset_y.
            Must be 1 or 0 (binary)
        dataset_x : first dimension of dataset
        dataset_y : second dimension of dataset (can be None, then the dataset is one dimension
    """
    plt.scatter(dataset_x,
                dataset_y if dataset_y is not None else [0 for _ in range(dataset_x.size()[0])],
                color=['b' if label < 0.5 else "r" for label in labels])
    plt.title(title)
    plt.show()

def plot_decision_boundary(dataset, labels, model):
    """
        plots decision boundary for binary classifier
        dataset : should be in two dimensions. Dataset points
        labels : labels of the dataset (should be binary)
    """
    color_map = plt.get_cmap("Paired")
    # Define region of interest by data limits
    xmin, xmax = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    ymin, ymax = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    x_span = np.linspace(xmin, xmax, 1000)
    y_span = np.linspace(ymin, ymax, 1000)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    model.eval()
    labels_predicted = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))

    # Plot decision boundary in region of interest
    labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.detach().numpy()]
    z = np.array(labels_predicted).reshape(xx.shape)
 
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=color_map, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels_predicted = model(dataset)
    ax.scatter(dataset[:, 0], dataset[:, 1],
               c=labels.reshape(labels.size()[0]), cmap=color_map, lw=0)
    plt.show()
    return fig, ax

def create_dataset(size_of_class_dataset=2000,
                   number_of_dim=2,
                   spacial_offset=10,
                   complicated=False):
    """
    dataset and labels creation and randomation
    size_of_class_dataset : size of one class in the dataset.
        The total dataset size will be 2 * size_of_class_dataset
    spacial_offset : the more it is, the easier it will be to classify data
    complicated : if True, adds a cluster to make this binary dataset non linearly separable
    """
    dataset_class_a = torch.randn(size_of_class_dataset * \
            (2 if not complicated else 1), number_of_dim)
    train_dataset_class_a = dataset_class_a[size_of_class_dataset // \
            (1 if not complicated else 2):]
    test_dataset_class_a = dataset_class_a[:size_of_class_dataset // \
            (1 if not complicated else 2)]

    # if we want a complicated dataset, we split class a to be at both side of class b,
    # therefore creating a non-linear classification problem
    if complicated:
        dataset_class_a_2 = torch.randn(size_of_class_dataset, number_of_dim) + spacial_offset * 2
        train_dataset_class_a_2 = dataset_class_a_2[size_of_class_dataset // 2:]
        test_dataset_class_a_2 = dataset_class_a_2[:size_of_class_dataset // 2]
        train_dataset_class_a = torch.cat([train_dataset_class_a,
                                           train_dataset_class_a_2], dim=0)
        test_dataset_class_a = torch.cat([test_dataset_class_a,
                                          test_dataset_class_a_2], dim=0)

    dataset_class_b = torch.randn(size_of_class_dataset * 2, number_of_dim) + spacial_offset
    train_dataset_class_b = dataset_class_b[size_of_class_dataset:]
    test_dataset_class_b = dataset_class_b[:size_of_class_dataset]

    train_dataset = torch.cat([train_dataset_class_a, train_dataset_class_b], dim=0)
    test_dataset = torch.cat([test_dataset_class_a, test_dataset_class_b], dim=0)

    train_labels_class_a = torch.zeros(size_of_class_dataset, 1)
    test_labels_class_a = torch.zeros(size_of_class_dataset, 1)

    train_labels_class_b = torch.ones(size_of_class_dataset, 1)
    test_labels_class_b = torch.ones(size_of_class_dataset, 1)

    train_labels = torch.cat([train_labels_class_a, train_labels_class_b], dim=0)
    test_labels = torch.cat([test_labels_class_a, test_labels_class_b], dim=0)

    # https://stackoverflow.com/questions/44738273/torch-how-to-shuffle-a-tensor-by-its-rows
    train_random_indexes = torch.randperm(size_of_class_dataset * 2)
    train_dataset = train_dataset[train_random_indexes]
    train_labels = train_labels[train_random_indexes]

    test_random_indexes = torch.randperm(size_of_class_dataset * 2)
    test_dataset = test_dataset[test_random_indexes]
    test_labels = test_labels[test_random_indexes]

    plot_dataset("train dataset", train_labels,
                 train_dataset[:, 0], None if number_of_dim == 1 else train_dataset[:, 1])
    plot_dataset("test dataset", test_labels,
                 test_dataset[:, 0], None if number_of_dim == 1 else test_dataset[:, 1])

    return train_dataset, train_labels, test_dataset, test_labels

class NeuralNetwork(nn.Module):
    """
    the deep learning model
    """

    def __init__(self, number_of_dim=2, hidden_layer=True):
        """
            hidden_layer : if False, the neural network is only
                a perceptron and not a multi-layer perceptron.
                The output layer is used as the only layer in that case
        """
        super().__init__()
        if hidden_layer:
            self.fully_connected_1 = nn.Linear(number_of_dim, 2)
            self.relu = nn.ReLU()
        self.output_layer = nn.Linear(number_of_dim, 1)
        self.output_activation = nn.Sigmoid()
        self.hidden_layer = hidden_layer

    def forward(self, inputs):
        """
        forward pass
        """
        if self.hidden_layer:
            outputs = self.relu(self.fully_connected_1(inputs))
        else:
            outputs = inputs
        outputs = self.output_activation(self.output_layer(outputs))
        return outputs

def weights_init(model):
    """
    initialize the weights of the model
    """
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight.data)

def train_epoch(model, dataset, labels, optimizer, criterion):
    """
    function that trains the model for one epoch
    """
    model.train()
    losses = []
    batch_size = 50
    for batch_index in range(0, dataset.size(0), batch_size):
        dataset_batch = dataset[batch_index:batch_index + batch_size, :]
        labels_batch = labels[batch_index:batch_index + batch_size, :]
        dataset_batch = Variable(dataset_batch)
        labels_batch = Variable(labels_batch)

        optimizer.zero_grad()
        # (1) Forward
        labels_infered = model(dataset_batch)
        # (2) Compute diff
        loss = criterion(labels_infered, labels_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        optimizer.step()
        losses.append(loss.data.numpy())
    return losses

def main(number_of_dim=2,
         spacial_offset=5,
         size_of_class_dataset=1000,
         complicated=False,
         hidden_layer=False,
         learning_rate=0.005,
         epochs_number=1000):
    """
    main function
    """
    train_dataset, train_labels, test_dataset, _ = \
            create_dataset(size_of_class_dataset, number_of_dim, spacial_offset, complicated)
    model = NeuralNetwork(number_of_dim, hidden_layer=hidden_layer)
    model.apply(weights_init)
    opt = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    e_losses = []
    plot = True
    for epoch_index in range(epochs_number):
        model.eval()
        outputs = model(test_dataset)
        if epoch_index % (epochs_number / 5) == 0 and plot:
            plot_dataset(f"epoch {epoch_index}/{epochs_number}",
                         outputs, test_dataset[:, 0],
                         None if number_of_dim == 1 else test_dataset[:, 1])
        e_losses += train_epoch(model, train_dataset, train_labels, opt, criterion)
    plt.plot(e_losses)
    plt.show()
    if number_of_dim != 1:
        plot_decision_boundary(train_dataset, train_labels, model)

def parse_arguments(argv):
    if len(argv) <= 1:
        return []
    parser = argparse.ArgumentParser(description="Process script arguments")
    parser.add_argument("--number_of_dim", type=int, choices=[1, 2],
                        dest="number_of_dim", nargs='?', default=2,
                        help="Number of dimsenion of the dataset (1 or 2)")
    parser.add_argument("--spacial_offset", type=int,
                        dest="spacial_offset", nargs='?', default=5,
                        help="Tells the distance between each cluster of data in the dataset")
    parser.add_argument("--size_of_class_dataset", type=int,
                        dest="size_of_class_dataset", nargs='?', default=1000,
                        help="Tells the number of point in one class. Given that there is " + \
                             "two class, the size of the dataset is size_of_class_dataset * 2")
    parser.add_argument("--complicated",
                        dest="complicated", action="store_true",
                        help="If True, adds a cluster to make this a non-linearly " + \
                             "separable problem")
    parser.add_argument("--hidden_layer",
                        dest="hidden_layer", action="store_true",
                        help="If True, the neural networks becomes a multi-layer perceptron." + \
                             "If False, the neural network is composed of a single neurone")
    parser.add_argument("--learning_rate", type=float, nargs='?', default=0.005,
                        dest="learning_rate",
                        help="Learning rate of the gradient descent algorithm")
    parser.add_argument("--epochs_number", type=int, nargs='?', default=500,
                        dest="epochs_number",
                        help="Number of epochs of the training phase")
    args = parser.parse_args()
    return [args.number_of_dim, args.spacial_offset, args.size_of_class_dataset, args.complicated,
            args.hidden_layer, args.learning_rate, args.epochs_number]

if __name__ == "__main__":
    args = parse_arguments(sys.argv)
    main(*args)
