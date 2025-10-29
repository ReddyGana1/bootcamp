from keras.models import Sequential
from keras.layers import Layers,Dense,Flatten
from keras.datasets import fashion_mnist,cifar100
from keras.utils import to_categorical
from keras.optimizers import adam
from keras import regularizers

(x_train,y_train),(x_test,y_test)=cifar100.load_data()

#normalize
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#to_categorical
y_train =to_categorical(y_train)
y_test =to_categorical(y_test)


#arch
model_base=Sequential()
model_base.add(Flatten(input_shape=(32,32,3)))
model_base.add(Dense(1024,activation='relu'))
model_base.add(Dense(512,activation='relu'))
model_base.add(Dense(256,activation='relu'))
model_base.add(Dense(128,activation='relu'))
model_base.add(Dense(64,activation='relu'))
model_base.add(Dense(100,activation='softmax'))

#compile 
model_base.complie(optimizer=adam(learning_rate=0.001,loss='categorical_crossentropy',metrics=['accuracy']))

#train
history=model_base.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

#evaluate
loss,test_accuray=model_base.evaluate(x_test,y_test)


#*************************

#model 2 with regularizer(le-14) and dropout
model_le4=Sequential()
#model_le4.add(Flatten(input_shape=(32,32,3)))
model_le4.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(le-4)))
model_le4.add(Dense(512,activation='relu'))
model_le4.add(Dense(256,activation='relu'))
model_le4.add(Dense(128,activation='relu'))
model_le4.add(Dense(64,activation='relu'))
model_le4.add(Dense(100,activation='softmax'))

#compile 
model_le4.complie(optimizer=adam(learning_rate=0.001,loss='categorical_crossentropy',metrics=['accuracy']))

#train
history=model_le4.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

#evaluate
loss,test_accuray=model_le4.evaluate(x_test,y_test)

