import tensorflow as tf
import sys
from tensorflow.keras.models import Sequential, Model, clone_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Dense, AveragePooling2D, add, GlobalAveragePooling2D # type: ignore
from tensorflow.keras import layers # type: ignore
from copy import deepcopy

def get_model(model:callable, data_shape: tuple[int, int, int], num_classes:int) -> Model:
    inputs = tf.keras.Input(shape=data_shape, dtype='float32', name="Input")
    ret_model = model(num_classes = num_classes)
    return tf.keras.Model(inputs=inputs, outputs=ret_model(inputs), name=model.__name__)
    
class BasicBlock(Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.do1 = layers.Dropout(0.1)
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.do2 = layers.Dropout(0.1)
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization(),
                layers.Dropout(0.1)
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.do1(self.bn1(self.conv1(x))))
        out = self.do2(self.bn2(self.conv2(out)))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BottleNeck(Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BuildResNet(Model):
    def __init__(self, block, num_blocks, num_classes):
        super(BuildResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        # print(len(layer))
        return tf.keras.Sequential(layer)
        # return layer[0]

def resnet18(num_classes):
    return BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes)
def resnet34(num_classes):
    return BuildResNet(BasicBlock, [3, 4, 6, 3], num_classes)
def resnet50(num_classes):
    return BuildResNet(BottleNeck, [3, 4, 6, 3], num_classes)
def resnet101(num_classes):
    return BuildResNet(BottleNeck, [3, 4, 23, 3], num_classes)
def resnet152(num_classes):
    return BuildResNet(BottleNeck, [3, 8, 36, 3], num_classes)

# #* End of ResNet Implementation 
class LeNet5(Model):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = layers.Conv2D(6, kernel_size=5, activation='relu', padding='same')
        self.pool1 = layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = layers.Conv2D(16, kernel_size=5, activation='relu')
        self.pool2 = layers.AveragePooling2D(pool_size=2, strides=2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='relu')
        self.fc2 = layers.Dense(84, activation='relu')
        self.fc3 = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def lenet5(num_classes=10):
    return LeNet5(num_classes=num_classes)

def average_model_weights(models, model_type, end, input_shape):

    workers = len(models)
    initial_weights = models[0].get_weights()
    
    sum_weights = initial_weights
    for model in models[1:]:
        model_weights = model.get_weights()
        sum_weights = tf.nest.map_structure(lambda x, y: x + y, sum_weights, model_weights)
    
    average_weights = tf.nest.map_structure(lambda x: tf.divide(x, workers), sum_weights)
    
    if model_type == 'lenet5': new_model = get_model(lenet5, input_shape, end)
    elif model_type == 'resnet18': new_model = get_model(resnet18, input_shape, end)    
    new_model.build((None, ) + input_shape)
    new_model.set_weights(average_weights)

    return new_model