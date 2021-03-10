from basic_structure import Placeholder
import numpy as np




if __name__ == "__main__":
    x_, y_, w0_, b0_, w1_, b1_ = np.random.normal(size=(1, 6)).squeeze()
    x = Placeholder(name='x', istrainable=False)
    y = Placeholder(name='y', istrainable=False)
    w0 = Placeholder(name='w0')
    b0 = Placeholder(name='b0')

    linear = Linear()

    feed_input = {'x': x,
                  'y': y,
                  'w0': w0,
                  'b0': b0,
                  'w1': w1,
                  'b1': b1,
                  }