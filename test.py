import numpy as np
import passengers as psg

from lstm import LstmParam, LstmNetwork

FILE = "D:/tmp/international-airline-passengers.csv"

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)
    data = psg.load_data(FILE)
    X, Y = psg.create_dataset(data, 7)
    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 14
    x_dim = 7
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    y_list = Y[:20]
    print(y_list)
    input_val_arr = X[:20]

    # y_list = [-0.5, 0.2, 0.1, -0.5]
    # input_val_arr = [np.random.random(x_dim) for _ in y_list] # a list of array which shape is (x_dim,)

    for cur_iter in range(10000):
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

        if cur_iter % 500 == 0:
            print("iter", "%2s" % str(cur_iter), end=": ")
            print("y_pred = [" +
              ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")
            print("loss:", "%.3e" % loss)

if __name__ == "__main__":
    example_0()


