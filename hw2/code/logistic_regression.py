import numpy as np
import matplotlib.pyplot as plt

# NOTE: The things to complete are marked by "##### ADD YOUR CODE HERE".

""" PLEASE use this completed function for for plot generation,
    for consistency among submitted plots. """


def plot(train_losses, train_accs, test_losses, test_accs, prefix):
    num_epochs = len(train_losses)
    plt.plot(range(num_epochs), train_losses, label='train loss', marker='o', linestyle='dashed', linewidth=1,
             markersize=2)
    plt.plot(range(num_epochs), test_losses, label='test loss', marker='o', linestyle='dashed', linewidth=1,
             markersize=2)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('log loss')
    plt.savefig('./{}_loss.pdf'.format(prefix))
    plt.clf()
    plt.plot(range(num_epochs), train_accs, label='train acc', marker='o', linestyle='dashed', linewidth=1,
             markersize=2)
    plt.plot(range(num_epochs), test_accs, label='test acc', marker='o', linestyle='dashed', linewidth=1, markersize=2)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('./{}_acc.pdf'.format(prefix))


""" This function should add repeated features,
    as needed for Part 3b."""


def augment(xs, vocab_idx, num_repeats):
    ##### ADD YOUR CODE HERE
    augmented_xs = np.zeros((xs.shape[0], xs.shape[1]+num_repeats))
    repeat_feature = xs[:, vocab_idx].reshape((-1, 1))
    augmented_xs[:, xs.shape[1]:] = np.tile(repeat_feature, (1, num_repeats))
    augmented_xs[:, :xs.shape[1]] = xs
    return augmented_xs


""" This function computes the logistic loss on (xs, ys).
    Please compute the MEAN over the dataset vs. the SUM,
    the latter of which was used in the preceding theory
    section."""


def loss(params, xs, ys):
    ##### ADD YOUR CODE HERE
    w, b = params
    n = xs.shape[0]
    linear_out = np.dot(xs, w) + b
    sigmoid_out = 1 / (1 + np.exp(-linear_out))
    return 1 / n * (-np.dot(ys.T, np.log(sigmoid_out)) - np.dot(1 - ys.T, np.log(1 - sigmoid_out)))[0, 0]


""" This function computes the accuracy on (xs, ys)."""


def acc(params, xs, ys):
    ##### ADD YOUR CODE HERE
    w, b = params
    n = xs.shape[0]
    linear_out = np.dot(xs, w) + b
    sigmoid_out = 1 / (1 + np.exp(-linear_out))
    num_correct = np.where(ys[sigmoid_out > 0.5] == 1)[0].shape[0] + np.where(ys[sigmoid_out <= 0.5] == 0)[0].shape[0]
    return num_correct / n


""" This function should compute the gradient of the
    logistic loss with respect to the parameters.
    The loss should be a MEAN over the dataset vs. the SUM,
    the latter of which was used in the preceding theory
    section.
    To ensure your code runs in a reasonable amount of time,
    it's important to vectorize this computation. If you're
    not sure what this means, please come to office hours. """


def get_gradients(params, xs, ys):
    ##### ADD YOUR CODE HERE
    w, b = params
    n = xs.shape[0]
    linear_out = np.dot(xs, w) + b
    sigmoid_out = 1 / (1 + np.exp(-linear_out))
    grad_w, grad_b = 1 / n * np.dot(xs.T, sigmoid_out - ys), 1 / n * np.sum(sigmoid_out - ys)
    return grad_w, grad_b


""" This function applies a gradient descent update
    step to params, using the provided computed gradients
    and learning_rate, and returns the new params."""


def apply_gradients(params, gradients, learning_rate):
    ##### ADD YOUR CODE HERE
    w, b = params
    grad_w, grad_b = gradients
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    return w, b


""" Run this function for Parts 1 & 2.
    This function runs gradient descent on (train_xs, train_ys)
    for num_epochs, using the specified learning_rate.
    The inputs train_xs, train_ys, test_xs, test_ys are obtained
    directly from the HW 1/HW 2 dataset dictionary.
    The labels are converted to {0,1} at the start of the function.
    The parameters are initialized at 0.
    The final metrics, along with parameters, are returned.
    YOU DO NOT NEED TO MODIFY ANY PARTS OF THIS FUNCTION. """


def train(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate):
    # Convert y to {0,1}. Map 1-->1 and 2-->0. Recall that original labels are in {1,2}.
    for i in range(len(train_ys)): train_ys[i, 0] = 1 if train_ys[i, 0] == 1 else 0
    for i in range(len(test_ys)): test_ys[i, 0] = 1 if test_ys[i, 0] == 1 else 0

    # Initialize w and b at 0.
    d = train_xs.shape[1]
    w = np.zeros((d, 1))
    b = 0.0

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in range(num_epochs):
        # Perform an update step.
        gradients = get_gradients((w, b), train_xs, train_ys)
        w, b = apply_gradients((w, b), gradients, learning_rate)
        # Compute and store train and test metrics.
        train_losses.append(loss((w, b), train_xs, train_ys))
        train_accs.append(acc((w, b), train_xs, train_ys))
        test_losses.append(loss((w, b), test_xs, test_ys))
        test_accs.append(acc((w, b), test_xs, test_ys))

    return train_losses, test_losses, train_accs, test_accs, w, b


""" Run this function for Part 3a - passing in the appropriate arguments.
    You will first need to complete the missing pieces above.
    YOU DO NOT NEED TO MODIFY ANY PARTS OF THIS FUNCTION. """


def q3a(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate, num_features):
    train_xs = train_xs[:, :num_features]
    test_xs = test_xs[:, :num_features]
    return train(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate)


""" Run this function for Part 3b - passing in the appropriate arguments.
    You will first need to complete the missing pieces above.
    YOU DO NOT NEED TO MODIFY ANY PARTS OF THIS FUNCTION. """


def q3b(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate, num_features, repeat_idx, num_repeats):
    train_xs = train_xs[:, :num_features]
    test_xs = test_xs[:, :num_features]
    train_xs = augment(train_xs, repeat_idx, num_repeats)
    test_xs = augment(test_xs, repeat_idx, num_repeats)
    return train(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate)


if __name__ == '__main__':
    import pickle

    with open("hw2data_TA.pkl", "rb") as f:
        data = pickle.load(f)

    XTrain = data["XTrain"]
    yTrain = data["yTrain"]

    XTest = data["XTest"]
    yTest = data["yTest"]

    Vocab = data["Vocabulary"]

    # Q4_1
    train_losses, test_losses, train_accs, test_accs, w, b = train(XTrain, yTrain, XTest, yTest, 100, 0.1)
    print(train_accs[-1], test_accs[-1])
    plot(train_losses, train_accs, test_losses, test_accs, "Q4_1")

    # Q4_2
    train_losses, test_losses, train_accs, test_accs, w, b = train(XTrain, yTrain, XTest, yTest, 3000, 0.1)
    print(train_accs[-1], test_accs[-1])
    plot(train_losses, train_accs, test_losses, test_accs, "Q4_2")

    # Q4_3a
    train_losses, test_losses, train_accs, test_accs, w, b = q3a(XTrain, yTrain, XTest, yTest, 5000, 0.1, 10)
    print(train_accs[-1], test_accs[-1])

    # Q4_3b
    train_losses, test_losses, train_accs, test_accs, w, b = q3b(XTrain, yTrain, XTest, yTest, 5000, 0.1, 10, 4, 500)
    print(train_accs[-1], test_accs[-1])

