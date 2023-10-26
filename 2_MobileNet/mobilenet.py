from tensorflow.keras.layers import DepthwiseConv1D
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.layers import ReLU, AveragePooling1D, Flatten, Dense
from tensorflow.keras import Model, Input
class MobileNetBlock(Model):
    def __init__(self, k, s, out_planes, name):
        super(MobileNetBlock, self).__init__(name=name)
        self.conv_dw = DepthwiseConv1D(kernel_size=k, strides=s, padding="same")
        self.bn_dw = BatchNormalization()
        self.act_dw = ReLU()

        self.conv_pw = Conv1D(filters=out_planes, kernel_size=1, strides=1)
        self.bn_pw = BatchNormalization()
        self.act_pw = ReLU()

    def call(self, input_tens):
        x = self.conv_dw(input_tens)
        x = self.bn_dw(x)
        x = self.act_dw(x)

        x = self.conv_pw(x)
        x = self.bn_pw(x)
        x = self.act_pw(x)
        return x

class MobileNet(Model):
    def __init__(self, mobile_block, **params):
        super(MobileNet, self).__init__()
        self.conv1 = Conv1D(filters=params["init_filters"], kernel_size=params["init_k"], strides=params["init_s"], padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = ReLU()

        self.mobile_layers = []
        for i in range(len(params["k_list"])):
            self.mobile_layers.append(mobile_block(params["k_list"][i], params["s_list"][i], int(params["f_list"][i]*params["alpha"]), f"mobile_{i}"))
        
        self.avgpool = AveragePooling1D(pool_size=params["pool_size"], strides=1,data_format='channels_last')
        self.flatten = Flatten()
        self.dense = Dense(params["num_class"], activation='softmax')

    def call(self, input_tens):
        x = self.conv1(input_tens)
        x = self.bn1(x)
        x = self.act1(x)

        for layer in self.mobile_layers:
            x = layer(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x


    def summary_(self, input_length):
        x = Input(shape=(input_length,1))
        model = Model(inputs=[x], outputs=self.call(x))
        print(model.summary())

def build_network(**params):
    model = MobileNet(MobileNetBlock, **params)  
    model.summary_(params["input_length"])
    return model
              
