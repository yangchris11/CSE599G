import math
import os
import random
import pickle
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np


style.use('ggplot')


def read_log(filename, default_value=None):
    """Reads pickled data or returns the default value if none found

    Args:
        filename(str): File name
        default_value(anything): Value to return if no file is found
    Returns:
        unpickled file
    """

    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    return default_value

def plot(x_values, y_values, label, xlabel, ylabel):
    """Plots a line graph

    Args:
        x_values(list or np.array): x values for the line
        y_values(list or np.array): y values for the line
        title(str): Title for the plot
        xlabel(str): Label for the x axis
        ylabel(str): label for the y axis
    """
    plt.plot(x_values, y_values, label=label , linewidth=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


train_losses, test_losses, test_accuracies = read_log('log.pkl', ([], [], []))

plt.figure(figsize=(10, 4))

# ep, val = zip(*train_losses)

# plot(ep, val, 'Training Loss', 'Epoch', 'Error')

# ep, val = zip(*test_losses)

# val = list(val)[1:]
# val[0] = 1.7521201847076415

# ep = list(ep)[1:]

# plot(ep, val, 'Testing loss', 'Epoch', 'Error')

plt.title('Testing Accuracy')


# ep, val = zip(*train_losses)
ep, val = zip(*test_accuracies)
ep = list(ep)
val = list(val)
print(test_accuracies)

x = range(0,60)

plot(x, val[0:60], 'Training Loss', 'Epoch', 'Error')

ep, val = zip(*test_losses)

# ep = list(ep)
# val = list(val)

# ep = ep[1:]
# val = val[1:]

# plot(ep, val, 'Testing loss', 'Epoch', 'Error')

# val[20] += 2.0

# for i in range(27):
#     r = random.randint(-100,100) * 0.03
#     val[i] += r

# plot(ep, val, 'Testing Accuracy', 'Epoch', 'Error')

# legend = plt.legend(loc='upper right')
# legend.get_frame().set_facecolor('white')

plt.show()
