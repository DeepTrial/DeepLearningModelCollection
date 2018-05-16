import keras,os
from keras.models import Model
from keras.layers.merge import add,multiply
from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,BatchNormalization,concatenate,Activation,AveragePooling2D
from keras import backend as K
from keras.layers.core import Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

from perception.bases.model_base import ModelBase

class SegmentionModel(ModelBase):
	def __init__(self,config=None):
		super(SegmentionModel, self).__init__(config)

		self.patch_height=config.patch_height
		self.patch_width = config.patch_width
		self.num_seg_class=config.seg_num

		self.build_model()
		self.save()

	def expend_as(self,tensor, axs,rep):
		my_repeat = Lambda(lambda x,axs,repnum: K.repeat_elements(x, repnum, axis=axs), arguments={'axs':axs,'repnum': rep})(tensor)
		return my_repeat

	def _MiniUnet(self,input,shape):
		x1 = Conv2D(shape, (3, 3), strides=(1, 1), padding="same",activation="relu")(input)
		pool1=MaxPooling2D(pool_size=(2, 2))(x1)

		x2 = Conv2D(shape*2, (3, 3), strides=(1, 1), padding="same",activation="relu")(pool1)
		pool2 = MaxPooling2D(pool_size=(2, 2))(x2)

		x3 = Conv2D(shape * 3, (3, 3), strides=(1, 1), padding="same",activation="relu")(pool2)

		x=concatenate([UpSampling2D(size=(2,2))(x3),x2],axis=3)
		x = Conv2D(shape*2, (3, 3), strides=(1, 1), padding="same",activation="relu")(x)

		x = concatenate([UpSampling2D(size=(2, 2))(x),x1],axis=3)
		x = Conv2D(shape, (3, 3), strides=(1, 1), padding="same", activation="sigmoid")(x)
		return x

	def _AttnBlock(self,input,shape):
		inputshape = K.int_shape(input)
		x = Conv2D(shape, (3, 3), strides=(1, 1), padding="same")(input)
		x = BatchNormalization()(x)
		x = Conv2D(shape, (3, 3), strides=(1, 1), padding="same")(x)
		x = BatchNormalization()(x)
		x = Conv2D(shape, (3, 3), strides=(1, 1), padding="same")(x)

		if inputshape[3] != shape:
			shortcut = Conv2D(shape, (1, 1), padding='same')(input)
		else:
			shortcut = input

		result = add([x, shortcut])
		result = Activation('relu')(result)

		attn_x=self._MiniUnet(input,shape)
		attn_patch = multiply([attn_x,result])

		attn_patch = add([attn_patch,x])

		return attn_patch

	def _ResBlock(self,input,shape,arate=1):
		inputshape = K.int_shape(input)

		x = Conv2D(shape,(3,3),strides=(1,1),padding="same",dilation_rate=(arate,arate))(input)
		x = BatchNormalization()(x)
		x = Conv2D(shape, (3, 3), strides=(1, 1), padding="same", dilation_rate=(arate, arate))(x)
		x = BatchNormalization()(x)
		x = Conv2D(shape, (3, 3), strides=(1, 1), padding="same", dilation_rate=(arate, arate))(x)

		if inputshape[3] != shape:
			shortcut = Conv2D(shape, (1, 1), padding='same')(input)
		else:
			shortcut = input

		result = add([x, shortcut])
		result = Activation('relu')(result)

		return result

	def _aspp(self,input):
		inputshape = K.int_shape(input)
		shape=inputshape[3]

		path1=Conv2D(128,(1,1),strides=(1,1),padding="same")(input)
		path1 = BatchNormalization()(path1)
		path1 = Activation('relu')(path1)
		shapepath1 = K.int_shape(path1)

		path2=Conv2D(128,(3,3),strides=(1,1),padding="same",dilation_rate=(6,6))(input)
		path2 = BatchNormalization()(path2)
		path2 = Activation('relu')(path2)

		path3=AveragePooling2D((16,16),padding="same")(input)
		path3_2 = BatchNormalization()(path3)
		path3_2 = Activation('relu')(path3_2)
		shapepath3=K.int_shape(path3_2)
		path3_3=Conv2DTranspose(128,(3,3),strides=(shapepath1[1]//shapepath3[1], shapepath1[2]//shapepath3[2]),activation="relu", padding="same")(path3_2)

		result=concatenate([path1,path2,path3_3],axis=3)
		result=Conv2D(128,(1,1),strides=(1,1),padding="same")(result)
		result = BatchNormalization()(result)
		result = Activation('relu')(result)
		return result

	def build_model(self):
		inputs = Input((self.patch_height, self.patch_width, 3))

		block1=self._ResBlock(inputs,32)   #128 128 32
		pool1=MaxPooling2D(strides=(2,2))(block1)

		block2 = self._ResBlock(pool1, 32)   # 64 64 32
		pool2 = MaxPooling2D(strides=(2,2))(block2)

		block3 = self._ResBlock(pool2, 64)  # 32 32 64
		pool3 = MaxPooling2D(strides=(2,2))(block3)

		block4_1 = self._ResBlock(pool3, 128, 1)  #16 16 128
		block4_2 = self._ResBlock(block4_1,128, 2)
		block4 = self._ResBlock(block4_2, 128, 4)

		aspp=self._aspp(block4)


		decode_16=Conv2DTranspose(64,(3,3),strides=(4,4),activation="relu",padding="same")(block4)
		decode_32 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu", padding="same")(block3)
		up1=concatenate([block2,decode_16,decode_32],axis=3)

		aspp = Conv2DTranspose(32, (3, 3), strides=(4, 4), activation="relu", padding="same")(aspp)
		up2= concatenate([aspp,up1],axis=3)

		decode=Conv2D(32,(3,3),strides=(1,1), padding="same")(up2)
		decode = BatchNormalization()(decode)
		decode = Activation('relu')(decode)

		decode2=Conv2DTranspose(32,(3,3),strides=(2,2),activation="relu",padding="same")(decode)

		decode3=concatenate([decode2,block1],axis=3)

		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(decode3)
		conv8 = core.Reshape((self.patch_height * self.patch_width, (self.num_seg_class + 1)))(conv8)
		############
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		plot_model(model, to_file=os.path.join(self.config.checkpoint, "deeplabv3+.png"), show_shapes=True)
		self.model = model





