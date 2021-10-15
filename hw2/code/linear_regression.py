import numpy as np

# Template code for gradient descent, to make the implementation
# a bit more straightforward.
# Please fill in the missing pieces, and then feel free to
# call it as needed in the functions below.
def gradient_descent(X, y, lr, num_iters):
    losses = []
    n, d = X.shape
    w = np.zeros((d,1))
    for i in range(num_iters):
        grad = np.dot(X.T, np.dot(X, w) - y) ##### ADD YOUR CODE HERE
        w = w - lr * grad
        loss = np.dot((np.dot(X, w) - y).T, np.dot(X, w) - y) ##### ADD YOUR CODE HERE: f(w) = ...? (with lambda = 0)
        losses.append(loss)
    return losses, w


# Part 3.
def part_3():
    X = np.array([[1,1],[2,3],[3,3]])
    Y = np.array([[1],[3],[3]])    

    ##### ADD YOUR CODE FOR ALL PARTS OF 3 HERE
    # X = X.astype(np.float32)
    # Y = Y.astype(np.float32)
    
    print("Q3a")
    print(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y))

    print("Q3b")
    print(np.linalg.det(np.dot(X, X.T)))
    print(np.dot(X.T, np.linalg.inv(np.dot(X, X.T))))
    print(np.dot(np.dot(X.T, np.linalg.inv(np.dot(X, X.T))), Y))

    # print("Q3c")
    # for l in [1, 1e-3, 1e-5]:
    #     print(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(l, np.eye(2))), X.T), Y))
    
    # print("Q3d")
    # for l in [1, 1e-3, 1e-5]:
    #     print(np.dot(np.dot(X.T, np.linalg.inv(np.dot(X, X.T) + np.dot(l, np.eye(3)))), Y))

    losses, w = gradient_descent(X, Y, 0.01, 10000)
    print(w)


# Part 4.
def part_4():
    X = np.array([[1,1],[2,3],[3,3]]).T
    Y = np.array([[1],[3]])

    ##### ADD YOUR CODE FOR BOTH PARTS OF 4 HERE


# Part 5.
def part_5():
    X1 = np.array([[100,0.001],[200,0.001],[-200,0.0005]])
    X2 = np.array([[100,100],[200,100],[-200,100]])
    Y = np.array([[1],[1],[1]])

    ##### ADD YOUR CODE FOR ALL PARTS OF 5 HERE


part_3()

