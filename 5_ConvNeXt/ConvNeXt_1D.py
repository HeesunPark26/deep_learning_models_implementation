from tensorflow.keras.layers import Conv1D, LayerNormalization, Activation, GlobalAveragePooling1D, Dense
from tensorflow.keras import Model, Sequential, Input
import tensorflow as tf
class ConvNextBlock(Model):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super(ConvNextBlock, self).__init__()
        self.dwconv = Conv1D(dim, kernel_size=7, padding="same", groups=dim) # depthwise conv
        self.norm = LayerNormalization(epsilon=1e-6)
        self.pwconv1 = Conv1D(dim*4, kernel_size=1, padding="same") # pointwise conv
        self.act = Activation("relu")
        self.pwconv2 = Conv1D(dim, kernel_size=1, padding="same") # pointwise conv
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None
        # 우선 stochastic depth는 제외..
        
    def call(self, input_tense):
        x = self.dwconv(input_tense)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input_tense + x
        return x
        
        
class ConvNeXt(Model):
    def __init__(self, **params):
        super(ConvNeXt, self).__init__()
        num_classes = params["num_classes"]
        depths = params["depths"]
        dims = params["dims"]
        self.downsample_layers = []
        
        # stem
        stem = Sequential([
            Conv1D(dims[0], kernel_size=4, strides=4),
            LayerNormalization(epsilon=1e-6),
        ], name = 'stem')
        self.downsample_layers.append(stem)
        
        for i in range(len(depths)-1):
            downsample_layer = Sequential([
                LayerNormalization(epsilon=1e-6),
                Conv1D(dims[i+1], kernel_size=2, strides=2),
            ], f'downsample_{i+1}')
            self.downsample_layers.append(downsample_layer)
        
        self.stages = []
        for i in range(len(depths)):
            stage = Sequential([
                ConvNextBlock(dims[i]) for j in range(depths[i])
            ], name=f'stage_{i+1}')
            self.stages.append(stage)
        self.global_avg_pool = GlobalAveragePooling1D()
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dense = Dense(num_classes)
        
    def call(self, x):
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.global_avg_pool(x)
        x = self.norm(x)
        x = self.dense(x)
        return x
    
    def summary_(self, **params):
        x = Input(shape=(params["input_length"],1))
        model = Model(inputs=[x], outputs=self.call(x))
        print(model.summary())


def build_network(**params):
    model = ConvNeXt(**params)
    model.summary_(**params)
    return model
