from keras.applications import resnet50
import keras.backend as K
from keras import layers
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization, Add, Lambda
from keras.models import Model
import numpy as np


class BNConv2D():
    def __init__(self, filters_size, kernal_size, name, padding='valid', strides=(1, 1), dilation_rate=(1, 1),
                activation=None):
        self.filters_size = filters_size
        self.kernal_size = kernal_size
        self.padding = padding
        self.strides = strides
        self.name = name
        self.dilation_rate = dilation_rate
        self.activation = activation
        
    def __call__(self, value):
        
        value = BatchNormalization(name='bn_' + self.name)(value)
        conv2d = Conv2D(self.filters_size, self.kernal_size, padding=self.padding, strides=self.strides,
                            name=self.name, dilation_rate=self.dilation_rate, activation=self.activation)(value)
        
        return conv2d
    

def RFB_BLOCK_S(input_layer, layer_index,  in_planes, stride=None):
    if stride is None:
        strides = (1,1)
    else:
        strides = stride
        
    index = str(layer_index)
    
    inter_planes = in_planes // 4
    
    branch1x1a = BNConv2D(in_planes, (1, 1), padding='same', strides=(1, 1), name='conv_1x1_branch_a_' + index)(input_layer)
    
    branch1x1a_a = BNConv2D(inter_planes, (1, 1), strides=(1, 1), padding='same', name='conv_1x1_branch_a_a' + index)(branch1x1a)
    branch3x3a_a =  BNConv2D(inter_planes, (3, 3), padding='same', strides=strides, name='conv_3x3_branch_a_a' + index)(branch1x1a_a)
    
    branch1x3b =  BNConv2D(inter_planes, (1, 3), padding='same', strides=strides, name='conv_1x3_branch_b_' + index)(branch1x1a)
    branch3x3b =  BNConv2D(inter_planes, (3, 3), padding='same', strides=(1, 1),  dilation_rate=(3, 3), name='atrous_conv_3x3_branch_b_' + index)(branch1x3b)

    branch1x1c = BNConv2D(in_planes, (1, 1), padding='same', strides=(1, 1), name='conv_1x1_branch_c_' + index)(input_layer)
    branch3x1c =  BNConv2D(inter_planes, (3, 1), padding='same', strides=strides, name='conv_3x1_branch_c_' + index)(branch1x1c)
    branch3x3c =  BNConv2D(inter_planes, (3, 3), padding='same', strides=(1, 1),  dilation_rate=(3, 3), name='atrous_conv_3x3_branch_c_' + index)(branch3x1c)

    branch1x1d = BNConv2D(in_planes, (1, 1), padding='same', strides=(1, 1), name='conv_1x1_branch_d_' + index)(input_layer)
    branch3x3d =  BNConv2D(inter_planes//2, (1, 3), padding='same', strides=(1, 1), name='conv_3x3_branch_d_stack_a' + index)(branch1x1d)
    branch3x3d =  BNConv2D((inter_planes//4) * 3, (3, 1), padding='same', strides=strides, name='conv_3x3_branch_d_sback_b' + index)(branch3x3d)
    branch3x3d =  BNConv2D(inter_planes, (3, 3), padding='same', strides=(1, 1),  dilation_rate=(5, 5), name='atrous_conv_3x3_branch_d_' + index)(branch3x3d)
   
    if stride is None:
        shorcut_branch = input_layer
    else:
        shorcut_branch = BNConv2D(in_planes, (1, 1), padding='same', strides=strides, name='shorcut_branch' + index)(input_layer)

    concat_layer = Concatenate()([branch3x3a_a, branch3x3b, branch3x3c, branch3x3d])
    conv_1x1 = BNConv2D(in_planes, (1, 1), padding='same', strides=(1, 1), name='conv_1x1' + index, activation='relu')(concat_layer)
    
    return Add()([conv_1x1, shorcut_branch])


def RFB_BLOCK(input_layer, layer_index,  in_planes, stride=None):
    if stride is None:
        strides = (1,1)
    else:
        strides = stride
        
    index = str(layer_index)
    
    inter_planes = in_planes // 8
    
    branch1x1a = BNConv2D(in_planes, (1, 1), padding='same', strides=(1, 1), name='conv_1x1_branch_a_' + index)(input_layer)
    
    branch1x1a_a = BNConv2D(inter_planes, (1, 1), strides=(1, 1), padding='same', name='conv_1x1_branch_a_a' + index)(branch1x1a)
    branch3x3a_a =  BNConv2D(inter_planes, (3, 3), padding='same', strides=strides, name='conv_3x3_branch_a_a' + index)(branch1x1a_a)
    
    branch3x3b =  BNConv2D((inter_planes//2) * 3, (3, 3), padding='same', strides=strides, name='conv_3x3_branch_b_' + index)(branch1x1a)
    branch3x3b =  BNConv2D((inter_planes//2) * 3, (3, 3), padding='same', strides=(1, 1),  dilation_rate=(3, 3), name='atrous_conv_3x3_branch_b_' + index)(branch3x3b)

    branch1x1c = BNConv2D(in_planes, (1, 1), padding='same', strides=(1, 1), name='conv_1x1_branch_c_' + index)(input_layer)
    branch3x3c =  BNConv2D((inter_planes//2) * 3, (1, 3), padding='same', strides=(1, 1), name='conv_3x3_branch_c_stack_a' + index)(branch1x1c)
    branch3x3c =  BNConv2D((inter_planes//2) * 3, (3, 1), padding='same', strides=strides, name='conv_3x3_branch_c_sback_b' + index)(branch3x3c)
    branch3x3c =  BNConv2D(inter_planes, (3, 3), padding='same', strides=(1, 1),  dilation_rate=(5, 5), name='atrous_conv_3x3_branch_c_' + index)(branch3x3c)
   
    if stride is None:
        shorcut_branch = input_layer
    else:
        shorcut_branch = BNConv2D(in_planes, (1, 1), padding='same', strides=strides, name='shorcut_branch' + index)(input_layer)
    
    concat_layer = Concatenate()([branch3x3a_a, branch3x3b, branch3x3c])
    conv_1x1 = BNConv2D(in_planes, (1, 1), padding='same', strides=(1, 1), name='conv_1x1' + index, activation='relu')(concat_layer)
    
    return Add()([conv_1x1, shorcut_branch])

class RFBText:
    def __init__(self, model_weights_path=None):
        self.backbone = resnet50.ResNet50(include_top=False,
                                          weights=None,
                                          input_tensor=None,
                                          pooling=None,
                                          classes=1000)

        self.model_weights_path = model_weights_path                            


    def build(self):
        C5 =  self.backbone.get_layer('bn5c_branch2c').output
        C4 =  self.backbone.get_layer('bn4f_branch2c').output
        C3 =  self.backbone.get_layer('bn3d_branch2c').output
        C2 =  self.backbone.get_layer('bn2c_branch2c').output

        C5_UP = UpSampling2D((2, 2))(C5)
        C5_UP = BNConv2D(1024, (1, 1), padding='same', name='top_down_1_bottleneck')(C5_UP)

        add_1 = Add()([C5_UP, C4])

        C4_UP = UpSampling2D((2, 2))(add_1)
        C4_UP = BNConv2D(128, (3, 3), padding='same', name='top_down_2')(C4_UP)
        C4_UP = RFB_BLOCK(input_layer=C4_UP, layer_index=0,  in_planes=128, stride=None)
        C4_UP = BNConv2D(512, (1, 1), padding='same', name='top_down_2_bottleneck')(C4_UP)

        add_2 = Add()([C4_UP, C3])

        C3_UP = UpSampling2D((2, 2))(add_2)
        C3_UP = BNConv2D(64, (3, 3), padding='same', name='top_down_3')(C3_UP)
        C3_UP = RFB_BLOCK_S(input_layer=C3_UP, layer_index=1,  in_planes=64, stride=None)
        C3_UP = BNConv2D(256, (1, 1), padding='same', name='top_down_3_bottleneck')(C3_UP)

        add_3 = Add()([C3_UP, C2])

        C2_UP = UpSampling2D((2, 2))(add_3)
        C2_UP = BNConv2D(32, (3, 3), padding='same', name='top_down_4')(C2_UP)
        C2_UP = RFB_BLOCK_S(input_layer=C2_UP, layer_index=2,  in_planes=32, stride=(2,2))
        C2_UP = BNConv2D(128, (1, 1), padding='same', name='top_down_4_bottleneck')(C2_UP)

        geo_map =  Conv2D(4, (1, 1), padding='same', name='geo_map', activation='sigmoid')(C2_UP)
        angle_map =   Conv2D(1, (1, 1), padding='same', name='angle_map', activation='sigmoid')(C2_UP)
        
        geo_map = Lambda(lambda x: x * 512)(geo_map)
        angle_map = Lambda(lambda x: (x - 0.5)* np.pi/2)(angle_map)

        F_score =  Conv2D(1, (1, 1), padding='same', name='F_score', activation='sigmoid')(C2_UP)
        F_geometry = Concatenate(axis=-1, name='F_geometry')([geo_map, angle_map])

        return Model(self.backbone.input, [F_score, F_geometry])

    def __call__(self):
        model =  self.build()

        if self.model_weights_path is not None:
            print('Loading Model Weights')
            model.load_weights(self.model_weights_path)

        return model