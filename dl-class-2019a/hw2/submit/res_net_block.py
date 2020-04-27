from .. import *


class ResNetBlock(LayerUsingLayer):
    def __init__(self, conv_params, parent=None):
        super(ResNetBlock, self).__init__(parent)
        self.conv_layers = SequentialLayer([
                            ConvLayer(*conv_params), 
                            ReLULayer(), 
                            ConvLayer(*conv_params)], 
                            self.parent) 
        
        self.add_layer = AddLayer((self.conv_layers, self.parent))
        self.relu2 = ReLULayer(self.add_layer)

        assert not any([parent is None for parent in self.conv_layers.parents])
        assert not any([parent is None for parent in self.add_layer.parents])
        assert not any([parent is None for parent in self.relu2.parents])

    @property
    def final_layer(self):
        # TODO
        return self.relu2

    def forward(self, data):
        # TODO
        x = data
        conv_x = self.conv_layers(x)
        x = self.add_layer([conv_x, x])
        x = self.relu2(x)

        return x
