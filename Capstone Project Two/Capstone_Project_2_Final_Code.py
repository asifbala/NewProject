import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import decode_predictions
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

warnings.simplefilter("ignore", category=DeprecationWarning)

train_labels = '/Users/asifbala/Desktop/humpback-whale-identification.zip Folder/train.csv'

train_df = pd.read_csv(train_labels)

train_df = train_df[:6300]

#df_test = train_df.iloc[12000:25361]

#print(df_test.describe())

print(train_df.shape)

print(train_df.head())

print(train_df.describe())

print(train_df.info())

train_df_IdCount = train_df.groupby('Id')['Image'].count()

train_df_IdCount = train_df_IdCount .sort_values(ascending=False)

print(train_df_IdCount.head(30))

def image_prep(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 224, 224, 3))
    count = 0    
    for fig in data['Image']:
        img = image.load_img("/Users/asifbala/Desktop/humpback-whale-identification.zip Folder/"+dataset+"/"+fig, target_size=(224, 224, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1    
    return X_train

def label_prep(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y = onehot_encoded
    return y

X = image_prep(train_df, train_df.shape[0], "train")

X /= 255

#X_test = image_prep(df_test, df_test.shape[0], "train")

#X_test /= 255

print(X.shape)

print(type(X))
        
y = label_prep(train_df['Id'])

print(y)

#y_labels = np.argmax(y,axis=1)

#print(y[:5])

#print(y_labels[:5])

#y_test = label_prep(df_test['Id'])

#y_labels_test = np.argmax(y_test,axis=1)

#print(y_test)

#print(y_labels)

'''
def plot_images(images, cls_true):
    assert len(images) == len(cls_true)     
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        xlabel = "{}".format(cls_true[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

plot_images(X[:9],y_labels[:9])
'''

'''
model.add(Dense(y.shape[1], activation='softmax', name='sm'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

history = model.fit(X, y,validation_split=0.3, epochs=100, batch_size=100, verbose=1,callbacks=[early_stopping_monitor])

#features_train = model.predict(X)

#print(features_train)
'''

'''
model= Sequential()

model.add(Flatten())

model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))

model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

history = model.fit(x, y,validation_split=0.3, epochs=100, batch_size=100, verbose=1,callbacks=[early_stopping_monitor])
'''


#model = VGG16(weights='imagenet', include_top=False)

#model.add((Flatten()))

#print(model.summary())

'''
# predict the probability across all output classes
yhat = model.predict(X)
# convert the probabilities to class labels
label = decode_predictions(yhat)
print(label)
# retrieve the most likely result, e.g. highest probability
#label = label[0][0]
# print the classification
#print('%s (%.2f%%)' % (label[1], label[2]*100))
'''


#base_model = VGG16(weights='imagenet', include_top=False)

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)

#train_datagen.fit(X)

#d = train_datagen.flow(X, batch_size=32)

#print(type(d))

#features = base_model.predict_generator(d,steps=10)

#print(features.shape)

from keras.callbacks import Callback

class Histories(Callback):
    def on_train_begin(self,logs={}):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc'))

histories = Histories()

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = Dense(512, activation='relu')(x)
x = Flatten()(x)
predictions = Dense(y.shape[1], activation='softmax')(x)

for layer in base_model.layers:
    layer.trainable = False

model = Model(input=base_model.input, output=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(X, y,epochs=5,batch_size=256,callbacks=[histories],validation_split=0.2)

print(histories.losses)

print(histories.accuracies)

#print(histories.val_losses)

x = range(len(histories.losses))

y = histories.losses

c = pd.DataFrame({'batch_number':x,'loss':y})

c = c.set_index('batch_number')

print(c)

c.plot()

#plt.xlabel('batch_number')

plt.ylabel('loss')

plt.show()

x2 = range(len(histories.accuracies))

y2 = histories.accuracies

d = pd.DataFrame({'batch_number':x2,'accuracy':y2})

d = d.set_index('batch_number')

print(d)

d.plot()

#plt.xlabel('batch_number')

plt.ylabel('accuracy')

plt.show()

#model.fit_generator(train_datagen.flow(X, y, batch_size=32),
#                    steps_per_epoch=len(X) / 32, epochs=10)

#features = tf.convert_to_tensor(features)

#sess = tf.InteractiveSession()  

#print(features.eval())

#sess.close()

#x = features.reshape(10,25088)

#print(type(x))

#x = Flatten()(features)

#print(x.shape)

'''
model = VGG16(weights='imagenet', include_top=False)
print(model.summary())

yhat = model.predict(X)
'''
'''
from keras.models import load_model
from keras import applications
from keras import optimizers

base_model = VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(y.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.summary()

from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
epochs = 50

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
train_datagen.fit(X)


history = model.fit_generator(
    train_datagen.flow(X, y, batch_size=batch_size),
    steps_per_epoch=X.shape[0] // batch_size,
    epochs=epochs)
'''

'''
model = Sequential()

model.add(Flatten(input_shape=(7,7,512)))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.8))

model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()

history = model.fit_generator(d, y, steps_per_epoch=20, epochs=25)
'''

'''
model = Sequential()

model.add(Dense(512, activation='relu',input_shape=(7,7,512)))

model.add(Dropout(0.8))

model.add(Flatten())

model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()

history = model.fit(features, y, steps_per_epoch=25, epochs=5)
'''

'''
base_model = VGG16(weights='imagenet',input_shape=(224,224,3), include_top=False)
x = base_model.output
x = Flatten()(x)
x = Dense(512,activation='relu')(x)
preds = Dense(y.shape[1], activation='softmax', name='sm')(x)
model = Model(inputs=base_model.input,outputs=preds)

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary() 

history = model.fit(X, y,validation_split=0.3, epochs=100, batch_size=256, verbose=1)
'''


'''
base_model = VGG16(include_top=False)
x = base_model.output
x = Flatten()(x)
x = Dropout(0.4)(x)
# let's add two fully-connected layer
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
# and a final FC layer with 'softmax' activation since we are doing a multi-class problem 
predictions = Dense(y.shape[1], activation='softmax',name='sm')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
'''

#model.load_weights('saved_models/weights.vgg16_BN_finetuned.h5')

#checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.pre_trained_vgg16_v3.hdf5', 
 #                              verbose=1, save_best_only=True)

#history = model.fit(X, y,validation_split=0.3, epochs=100, batch_size=256, verbose=1)




'''
model.fit(X, y,
          batch_size=batch_size,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[histories]
         )
'''
'''
model = Sequential()

model.add(Conv2D(32, (3, 3), strides = (1, 1), padding='same', name = 'conv0',input_shape = (224, 224, 3)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), strides = (1,1),padding='same', name="conv1"))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))

model.add(Conv2D(32, (3, 3), strides = (1,1),padding='same', name="conv2"))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), strides = (1,1),padding='same', name="conv3"))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool2'))

model.add(Conv2D(32, (3, 3), strides = (1,1),padding='same', name="conv4"))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), strides = (1,1),padding='same', name="conv5"))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool3'))

model.add(Conv2D(32, (3, 3), strides = (1,1),padding='same', name="conv6"))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), strides = (1,1),padding='same', name="conv7"))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool4'))

model.add(Conv2D(64, (3, 3), strides = (1,1),padding='same', name="conv8"))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), strides = (1,1),padding='same', name="conv9"))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool5'))

model.add(Conv2D(64, (3, 3), strides = (1,1),padding='same', name="conv10"))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), strides = (1,1),padding='same', name="conv11"))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool6'))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

history = model.fit(X,y,validation_split=0.2, epochs=1, batch_size=32, verbose=1,callbacks=[histories])

print(histories.losses)

print(histories.accuracies)

x = range(len(histories.losses))

y = histories.losses

c = pd.DataFrame({'new':x,'new2':y})

print(c)

c = c.set_index('new')

print(c)

c.plot()

#plt.show()

#x= [4,5,6]

#y =[4,5,6]

#plt.plot(x,y)

#plt.show()

#plt.plot(range(len(histories.accuracies)),histories.accuracies)

#plt.show()
'''
'''
y_pred = model.predict_classes(X_test)

y_pred_proba = model.predict_proba(X_test)

print(y_pred)

print(y_pred_proba)
'''