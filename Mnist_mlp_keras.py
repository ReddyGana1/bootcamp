#import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#load mnist dataset
from keras.datasets import mnist
#load dataset
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print(f"label is {y_train[0]}")
#normalize the data
X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0
#to categorical encoding
print(f"Before one hot encoding:",y_train[100])
y_train=to_categorical(y_train)
print(f"After one hot encoding:",y_train[100])
y_test=to_categorical(y_test)
#architecture of model
model=Sequential()
model.add(Flatten(input_shape=(28,28))) #input layer
model.add(Dense(128,activation='relu')) #hidden layer 1
model.add(Dense(10,activation='softmax'))#output layer
#compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#train the model
result=model.fit(X_train,y_train,epochs=10,batch_size=64,validation_split=0.2)
#evaluate the model
loss,accuracy=model.evaluate(X_test,y_test)
print(f"Test accuracy:{accuracy}")
#make predictions
predictions=model.predict(X_test)
predicted_classes=np.argmax(predictions,axis=1)
true_classes=np.argmax(y_test,axis=1)   
print(f"Predicted classes:{predicted_classes[:10]}")
print(f"True classes:{true_classes[:10]}")
print("test loss:",loss)
print("test accuracy:",accuracy)
print(result.history.keys())
print(result.history['loss'])
print(result.history['accuracy'])
print(result.history['val_accuracy'])
print(result.history.values())
print(result.history)
plt.plot(result.history['val_accuracy'],label='valiation accuracy')
plt.plot(result.history['accuracy'],label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
#visualize some predictions
for i in range(5):
    plt.imshow(X_test[i],cmap='gray')
    plt.title(f"True label:{true_classes[i]},Predicted label:{predicted_classes[i]}")
    plt.show()
#make prediction on unseen data
unseen_images=X_test[:3]
unseen_predictions=model.predict(unseen_images)

