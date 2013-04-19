from scipy.linalg.decomp_schur import eps
from mnist import MNIST
import mlp
import numpy as np
import pylab as plt

testing_set = 10000      # max 10 000
hidden_layers = [200]

train = False
learning_set = 30000     # max 60 000
iterations = 500
tests_for_learning_curve = 1000

file_name = "weights/weights-l{0}-i{1}-h{2}*{3}".format(learning_set, iterations, len(hidden_layers), hidden_layers[0])

print "Neural network initialization..."
mpl = mlp.MPL(input_layer_size=28 ** 2, output_layer_size=10, hidden_layers_sizes=hidden_layers)

print "Loading testing data..."
mndata = MNIST('./mnist')
imgs_t, labels_t = mndata.load_testing()

if train:
    print "Loading training data..."
    mndata = MNIST('./mnist')
    imgs, labels = mndata.load_training()

    print "Preparing training data..."
    data = []
    for i in range(learning_set):
        label = [0] * labels[i] + [1] + [0] * (9 - labels[i])
        data.append((np.array(imgs[i]), np.array(label)))

    print "Learning neural network..."
    accuracy = np.zeros((2,iterations), dtype=int)
    for i in range(iterations):
        mpl.backpropagate(data, version=0)

        for j in range(tests_for_learning_curve):
            result =  np.argmax(mpl.compute(np.array(imgs[j])))
            if result == labels[j]: accuracy[0][i] += 1
            result =  np.argmax(mpl.compute(np.array(imgs_t[j])))
            if result == labels_t[j]: accuracy[1][i] += 1

        print iterations - i, " - training: {0:.2f}, testing {1:.3f}".format(
            accuracy[0][i] * 100.0 / tests_for_learning_curve, accuracy[1][i] * 100.0 / tests_for_learning_curve)

    np.save(file_name + ".npy", (mpl.weights))
    print "Done."
else:
    print "Loading weights from file..."
    mpl.weights = np.load(file_name + ".npy")

print "Testing testing data..."
hits = 0
for i in range(testing_set):
    result =  mpl.compute(np.array(imgs_t[i]))
    result = np.argmax(result)
    if result == labels_t[i]:
        hits += 1

result = "accuracy: {0:.3}%".format(hits * 100. / testing_set)
print result

if train:
    accuracy = accuracy * 100. / tests_for_learning_curve
    plt.plot(accuracy[0], "-r")
    plt.plot(accuracy[1], "-b")
    plt.title(result)
    plt.savefig(file_name + " - learning curve.png")
    plt.show()