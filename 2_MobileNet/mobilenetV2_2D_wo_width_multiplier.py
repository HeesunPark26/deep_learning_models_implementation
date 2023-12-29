import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, Add, BatchNormalization, AveragePooling2D, Dense
from tensorflow.keras import Model, Input, Sequential

class MobileNetV2Block(Model):
    def __init__(self, expansion_factor, stride, in_planes, out_planes, kernel):
        super(MobileNetV2Block, self).__init__()
        
        # stride should be either 1 or 2
        assert stride in [1, 2]
        
        # do residual connection only when stride==1 and in_planes == out_planes
        self.residual = (stride == 1 and in_planes==out_planes)
        
        ##### BUILD INVERTED RESIDUAL BLOCK #####
        self.layers_ = Sequential()
        
        ### --- Expansion layer: point-wise convolution
        # omit when expansion factor == 1
        if expansion_factor != 1:
            self.layers_.add(
                Sequential([
                    Conv2D(expansion_factor*in_planes, (1,1), padding='same', use_bias=False),
                    BatchNormalization(),
                    ReLU(6.0)
                ]))
        ### --- Depthwise convolution layer
        self.layers_.add(
            Sequential([
                DepthwiseConv2D(kernel, stride, padding="same", use_bias=False),
                BatchNormalization(),
                ReLU(6.0)
            ]))
        
        ### --- Projection layer: point-wise linear convolution
        self.layers_.add(
            Sequential([
                Conv2D(out_planes, (1,1), strides=1, padding="same", use_bias=False),
                BatchNormalization()
            ]))
        
    def call(self,input_tense):
        x = self.layers_(input_tense)
        if self.residual:
            x = Add()([x, input_tense])
        return x
    
class MobileNetV2Seq(Model):
    def __init__(self, t, n, s, in_planes, out_planes, kernel_size=3, name=None):
        super(MobileNetV2Seq, self).__init__(name=name)
        
        # for making repeated MobileNetV2Blocks
        self.blocks = Sequential()
        for i in range(n):
            stride = s if i==0 else 1
            self.blocks.add(MobileNetV2Block(t, stride, in_planes, out_planes, kernel_size))
            in_planes = out_planes
            
    def call(self, x):
        x = self.blocks(x)
        return x
                
class MobileNetV2(Model):
    def __init__(self, init_filters=32, inverted_residual_setting = None):
        super(MobileNetV2, self).__init__()
        ### --- the first conv layer 
        self.init_layer = Sequential([
            Conv2D(init_filters, (3,3), strides=(2,2), padding="same", use_bias=False),
            BatchNormalization(),
            ReLU(6)
        ])
        
        ##### BUILD INVERTED RESIDUAL BLOCK #####
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        self.mobileV2_seq = []
        in_planes = init_filters
        for t, c, n, s in inverted_residual_setting:
            self.mobileV2_seq.append(MobileNetV2Seq(t,n,s,in_planes,c))
            in_planes = c
            
        ### --- the last conv layer 
        self.last_layer = Sequential([
            Conv2D(1280, (1,1), strides=(1,1), padding="same", use_bias=False),
            BatchNormalization(),
            ReLU(6)
        ])
        
        ### --- pooling & classifier 
        self.avgpool = AveragePooling2D(pool_size=(7,7), padding="same")
        self.classifier = Dense(1000, activation='softmax', use_bias=False)

    def call(self, x):
        x = self.init_layer(x)
        for seq in self.mobileV2_seq:
            x = seq(x)
        x = self.last_layer(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
    
    def summary_(self):
        x = Input(shape=(224,224,3))
        model = Model(inputs = x, outputs = self.call(x))
        print(model.summary())
