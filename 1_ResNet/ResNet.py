### load modules
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Lambda, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras import Model, Input
import tensorflow as tf


# for decreasing output size and increasing filter number
def zeropad(x, times):
    y =  tf.zeros_like(x)
    if times > 2:
        y_new = tf.zeros_like(x)
        for _ in range(times-2):
            y_new = tf.concat([y_new,y], axis=3)
        y = y_new
    return tf.concat([x, y], axis=3)

def zeropad_output_shape(input_shape, out_planes):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[3] = out_planes
    return tuple(shape)

def make_shortcut(x, downsampling, out_planes):
    if downsampling:
        x = MaxPooling2D(pool_size=(2,2))(x)
    if x.shape[3] != out_planes:
        x = Lambda(zeropad, output_shape=zeropad_output_shape(x.shape, out_planes), arguments={"times":int(out_planes/x.shape[3])})(x)
    return x


### ResNet basic block
class ResBlock(Model):
    def __init__(self, out_planes, stride):
        super(ResBlock, self).__init__()
        
        self.downsampling = stride==2 
        self.out_planes = out_planes

        # layer 1
        self.conv1 = Conv2D(out_planes, (3,3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        
        # layer 2
        self.conv2 = Conv2D(out_planes, (3,3), padding='same')
        self.bn2 = BatchNormalization()
        
        # shortcut connection
        self.add = Add()
        self.act2 = Activation('relu')
    
    def call(self, input):
        shortcut = make_shortcut(input, self.downsampling, self.out_planes)
        # layer 1
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        
        # layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        # shortcut connection
        x = self.add([x, shortcut])
        x = self.act2(x)
        return x




### ResNet bottleneck block
class ResBottleneckBlock(Model):
    def __init__(self, out_planes, stride):
        super(ResBottleneckBlock, self).__init__()
        
        self.downsampling = stride==2 
        self.out_planes = out_planes
        # layer 1
        self.conv1 = Conv2D(out_planes//4, (1,1), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        
        # layer 2
        self.conv2 = Conv2D(out_planes//4, (3,3),  padding='same')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        
        # layer 3
        self.conv3 = Conv2D(out_planes, (1,1), padding='same')
        self.bn3 = BatchNormalization()
        
        # shortcut connection
        self.add = Add()
        self.act3 = Activation('relu')
    
    def call(self, input):
        shortcut = make_shortcut(input, self.downsampling, self.out_planes)        
        # layer 1
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        
        # layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        
        # shortcut connection
        x = self.add([x, shortcut])
        x = self.act3(x)
        
        return x
    
### Stacked Layer, containing multiple ResNet basic block or bottleneck block
class StackedLayer(Model):
    def __init__(self, out_planes, stride, num_blocks, use_bottleneck):
        super(StackedLayer, self).__init__()
        
        block = ResBottleneckBlock if use_bottleneck else ResBlock
        self.resblocks = tf.keras.Sequential([
            block(out_planes, stride if i == 0 else 1) for i in range(num_blocks)
        ])

    def call(self, x):
        return self.resblocks(x)

# ResNet34
class ResNet34(Model):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv1 = Conv2D(64, (7, 7), strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        
        self.pool1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')
        
        self.stack1 = StackedLayer(64, 1, 3, False)
        self.stack2 = StackedLayer(128, 2, 4, False)
        self.stack3 = StackedLayer(256, 2, 6, False)
        self.stack4 = StackedLayer(512, 2, 3, False)
        
        self.pool2 = GlobalAveragePooling2D()
        self.dense = Dense(1000)
        self.act2 = Activation('softmax')
        
    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        
        x = self.pool2(x)
        x = self.dense(x)
        x = self.act2(x)
        return x
    
    def summary_(self):
        x = Input(shape=(224, 224,1))
        model = Model(inputs=[x], outputs=self.call(x))
        print(model.summary())
        
        
         
            
# ResNet50
class ResNet50(Model):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = Conv2D(64, (7, 7), strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        
        self.pool1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')
        
        self.stack1 = StackedLayer(256, 1, 3, True)
        self.stack2 = StackedLayer(512, 2, 4, True)
        self.stack3 = StackedLayer(1024, 2, 6, True)
        self.stack4 = StackedLayer(2048, 2, 3, True)
        
        self.pool2 = GlobalAveragePooling2D()
        self.dense = Dense(1000)
        self.act2 = Activation('softmax')
        
    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        
        x = self.pool2(x)
        x = self.dense(x)
        x = self.act2(x)
        return x
    
    def summary_(self):
        x = Input(shape=(224, 224,1))
        model = Model(inputs=[x], outputs=self.call(x))
        print(model.summary())
        
            