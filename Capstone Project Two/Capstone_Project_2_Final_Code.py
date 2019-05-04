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
from keras.preprocessing import image
from keras.models import Model

warnings.simplefilter("ignore", category=DeprecationWarning)

train_labels = '/Users/asifbala/Desktop/humpback-whale-identification.zip Folder/train.csv'

train_df = pd.read_csv(train_labels)

df_train = train_df[:5000]

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

X = image_prep(df_train, df_train.shape[0], "train")

X /= 255

#X_test = image_prep(df_test, df_test.shape[0], "train")

#X_test /= 255
        
y = label_prep(df_train['Id'])

y_labels = np.argmax(y,axis=1)

print(y[:5])

print(y_labels[:5])

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

history = model.fit(X,y,validation_split=0.3, epochs=2, batch_size=7, verbose=1)

y_pred = model.predict_classes(X_test)

y_pred_proba = model.predict_proba(X_test)

print(y_pred)

print(y_pred_proba)
