class DenseUnit(Model):
    def __init__(self, out_planes, dropout=0):
        super(DenseUnit, self).__init__()
        # batch norm - relu - conv
        # bottleneck
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.conv1x1 = Conv2D(filters=out_planes*4, kernel_size=1)
        
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')
        self.conv3x3 = Conv2D(filters=out_planes, kernel_size=3, padding='same')
        
        self.concat = Concatenate()
        self.dropout = dropout
    def call(self, x): # x: batch_size, height, width, in_planes
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1x1(out)
        if self.dropout > 0:
            out = Dropout(self.dropout)(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3x3(out) # batch_size, height, width, out_planes
        if self.dropout > 0:
            out = Dropout(self.dropout)(out)
        
        return self.concat([x, out]) # batch_size, height, width, [in_planes + out_planes]

class DenseBlock(Model):
    def __init__(self, num_layers, growth_rate, dropout=0):
        super(DenseBlock, self).__init__()
        self.layers_list = []
        for _ in range(num_layers):
            self.layers_list.append(DenseUnit(growth_rate, dropout))
    def call(self, x):
        out = x
        for l in self.layers_list:
            out = l(out)
        return out
      
class TransitionLayer(Model):
    def __init__(self, out_planes, dropout):
        super(TransitionLayer, self).__init__()
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.conv1 = Conv2D(filters=out_planes, kernel_size=1, padding='same')
        self.pool1 = MaxPooling2D(pool_size=2, strides=2)
        self.dropout = dropout
    def call(self,x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        if self.dropout > 0:
            out = Dropout(self.dropout)(out)
        out = self.pool1(out)
        return out

class DenseNet(Model):
    def __init__(self, dropout, growth_rate, reduction_rate=0.5):
        super(DenseNet, self).__init__()
        self.conv1 = Conv2D(filters=16, kernel_size=7, strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        
        self.pool1 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        
        self.dense_, self.trans_ = [], []
        for num_layers in [6, 12, 24, 16]:
        
            self.dense_.append(DenseBlock(num_layers, growth_rate, dropout))
            self.trans_.append(TransitionLayer(int(growth_rate*num_layers*reduction_rate),dropout))
            
        self.pool_last = GlobalAveragePooling2D()
        self.fc = Dense(1000)
        self.act_last = Activation('softmax')
    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.pool1(out)
        
        for i in range(len(self.dense_)):
            out = self.dense_[i](out)
            if i < len(self.dense_)-1: # Don't apply transition layer on last block
                out = self.trans_[i](out)
        out = self.pool_last(out)
        out = self.fc(out)
        out = self.act_last(out)
        return out
        
