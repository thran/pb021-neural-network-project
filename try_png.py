import mlp
import numpy as np
from mnist import MNIST
from scipy import misc

weights_name = "weights/weights-l60000-i1000-h1*300.npy"
image_name =  "my numbers/8.png"

print "Neural network initialization..."
mpl = mlp.MPL(input_layer_size=28 ** 2, output_layer_size=10, hidden_layers_sizes=[200])
mpl.weights = np.load(weights_name)

if False:
    imgs, labels = MNIST('./mnist').load_testing()
    img = np.array(imgs[1224]).reshape(28,28)
    img[img>1] = 1

    print img

if True:
    image = misc.imread(image_name)
    image = 255 - np.mean(image, 2)
    img = np.zeros((28,28), dtype=int)
    img[image>10] = 255
    print img/255

    print np.argmax(mpl.compute(img.reshape((-1))))

