def layer(op):
    def layer_decorated(self):
        print("layer")
        op(self)
        return self
    return layer_decorated



class Network(object):
    def __init__(self):
        self.setup()

    def feed(self):
        print("feed")
        return self

    @layer
    def conv(self):
        print("conv")

    @layer
    def prelu(self):
        print("relu")

    @layer
    def max_pool(self):
        print("max_pool")

    @layer
    def softmax(self):
        print("softmax")

    @layer
    def fc(self):
        print("fc")

class RNet(Network):
    def setup(self):
        (self.feed()
         .conv()
         .prelu()
         .max_pool()
         .conv()
         .prelu()
         .max_pool()
         .conv()
         .prelu()
         .fc()
         .prelu()
         .fc()
         .softmax()
         )

rnet = RNet()
print("a")
         
