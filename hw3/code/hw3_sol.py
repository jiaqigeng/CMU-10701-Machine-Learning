
import numpy as np

from hw3_lib import Linear, Sigmoid, ReLU, CrossEntropyLoss, GradientDescentOptimizer


# NOTATION
# in the comments below
#     N is batch size.
#     in_features is the input size (not counting the bias)
#     out_features is the output size
#     C is number of classes (10 for this assignment)
#     E is the number of epochs

## do not modify these functions
def init_linear_params(in_size, out_size):
    stdv = 1 / np.sqrt(out_size)
    return (np.random.uniform(-stdv, stdv, (in_size, out_size)),
            np.random.uniform(-stdv, stdv, (1, out_size))
            )


class MultiLayerPerceptron:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, criterion):
        dx = criterion.backward()
        for layer in reversed(self.layers):
            dx = layer.backward(dx)


# this function will not be tested
# however, you should keep the default seed for the written portion of this question
def define_network(size_hidden,
                   size_input=784,
                   size_output=10,
                   seed=10701):
    """
    :param num_hidden: the number of hidden units
    :param seed: seed used to generate initial random weights. can be ignored.
    :param size_input: number of input features. 784 for this assignment
    :param size_output: number of output classes. 10 for this assignment
    """

    # do not change this line
    np.random.seed(seed)

    # complete model definition
    model = MultiLayerPerceptron([
        Linear(init_linear_params(size_input, size_hidden)),
        Sigmoid(),
        Linear(init_linear_params(size_hidden, size_output))
    ])

    return model


def train_network(model, dataset, num_epoch,
                  learning_rate, batch_size,
                  seed=10701):
    """
    :param model: neural network object
    :param dataset: a dictionary of following keys:
            'train_x': np.ndarray, shape (N, 784)
            'train_y': np.ndarray, shape (N, )
            'test_x': np.ndarray of int, shape (N_test, 784)
            'test_y': np.ndarray of int, shape (N_test, )

            for training_data_student, we should have
            N=3000, and N_test=1000

    :param num_epoch: (E) the number of epochs to train the network
    :param learning_rate: the learning_rate multiplied on the gradients
    :param seed: an integer used to generate random initial weights,
           not needed for autolab.

    :return: should be a dictionary containing following keys:

        'train_loss': list of training losses, its size should equal E


        'test_loss': list of testing losses, its size should equal E

        'train_accuracy': list of training accuracies, its size should equal E

        'test_accuracy': list of testing accuracies, its size should equal E

        'yhat_train': final list of prediction labels for training dataset,
                      its size should equal N

        'yhat_test': final list of prediction labels for testing dataset,
                     its size should equal N_test
    """
    # get data
    train_x, train_y = dataset['train_x'], dataset['train_y']
    test_x, test_y = dataset['test_x'], dataset['test_y']

    # do not change this line
    np.random.seed(seed)

    training_loss_all = []
    training_acc_all = []
    testing_loss_all = []
    testing_acc_all = []

    optimizer = GradientDescentOptimizer(model, learning_rate)
    criterion = CrossEntropyLoss()

    for idx_epoch in range(num_epoch):  # for each epoch
        for slice_start in range(0, len(train_x), batch_size):
            x = train_x[slice_start: slice_start + batch_size, :]
            y = train_y[slice_start: slice_start + batch_size]

            # forward
            y_pred = model.forward(x)

            # do not use this loss
            running_loss = criterion(y_pred, y)

            # backward
            model.backward(criterion)

            optimizer.step()

        # now we arrive at the end of this epoch,
        # we want to compute some statistics.

        # training_loss_this_epoch is average loss over all batches of training data.
        # note that you should compute this loss ONLY using the model
        # obtained at the END of this epoch,
        # i.e. you should NOT compute this loss ON THE FLY by averaging
        # intermediate training losses DURING this epoch.
        #
        ### TYPE HERE AND REMOVE `pass` below ###
        y_pred = model.forward(train_x)
        training_loss_this_epoch = criterion(y_pred, train_y)

        # record training loss
        # this float() is just there so that result can be JSONified easily
        training_loss_all.append(float(training_loss_this_epoch))

        # generate predicted labels for training data
        # yhat_train_all should be a 1d vector of same shape as train_y
        ### TYPE HERE AND REMOVE `pass` below ###
        yhat_train_all = np.argmax(criterion.y_hat, axis=1)

        # record training error
        training_acc_all.append(float((yhat_train_all == train_y).mean()))

        # testing_loss_this_epoch is average loss over all batches of test data.
        # use the same batch size as training
        ### TYPE HERE AND REMOVE `pass` below ###
        y_pred = model.forward(test_x)
        testing_loss_this_epoch = criterion(y_pred, test_y)

        # record testing loss
        testing_loss_all.append(float(testing_loss_this_epoch))

        # generate yhat for testing data
        ### TYPE HERE AND REMOVE `pass` below ###
        yhat_test_all = np.argmax(criterion.y_hat, axis=1)

        # record testing error
        testing_acc_all.append(float((yhat_test_all == test_y).mean()))

    # keep this part intact, do not modify it.
    return {
        # losses and accuracy across epochs.
        'train_loss': training_loss_all,
        'test_loss': testing_loss_all,
        'train_accuracy': training_acc_all,
        'test_accuracy': testing_acc_all,

        # yhat of the final model at the last epoch.
        # tolist for JSON.
        'yhat_train': yhat_train_all.tolist(),
        'yhat_test': yhat_test_all.tolist(),
    }


# how to generate dataset ready for `train_network`
# when you are testing your data.

def load_data():
    data_train = np.genfromtxt('training_data_student/train.csv', dtype=np.float64, delimiter=',')
    assert data_train.shape == (3000, 785)
    data_test = np.genfromtxt('training_data_student/test.csv', dtype=np.float64, delimiter=',')

    assert data_test.shape == (1000, 785)

    return {
        'train_x': data_train[:, :-1].astype(np.float64),
        'train_y': data_train[:, -1].astype(np.int64),
        'test_x': data_test[:, :-1].astype(np.float64),
        'test_y': data_test[:, -1].astype(np.int64),
    }


# # how to test your solution locally.
# np.random.seed(10701)
# model = MultiLayerPerceptron([
#     Linear(init_linear_params(784, 256)),
#     Sigmoid(),
#     Linear(init_linear_params(256, 10)),
# ])
#
# # model = define_network(size_input=784, size_hidden=256, size_output=10)
# dataset = load_data()
# result = train_network(model, dataset, num_epoch=50, batch_size=1, learning_rate=0.01)

