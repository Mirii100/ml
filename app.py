print('welcome to the neural network class tutorial')

# gradient descent - capable of learning linearly
# multilayer neural networks - has hidden layers
"""
playground.tensorflow.org 

overfitting problem - solved by dropout
"""

import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split 

# # reading data 
# with open("banknotes.csv") as f:
#     reader = csv.reader(f)
#     next(reader) 
    
#     data=[]
#     for row in reader:
#         if len(row) < 5:  # skip incomplete rows
#             continue
#         label_value = row[4].strip()    
#         data.append({
#         "evidence":[float(cell) for cell in row[:4]],
#         "label": 1 if row[4].strip() == "1" else 0
#                 })


with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)  # try to skip header

    data = []
    for row in reader:
        # Skip empty or bad rows
        if len(row) < 5:
            continue
        try:
            evidence = [float(cell) for cell in row[:4]]
            label = 1 if row[4].strip() == "1" else 0
            data.append({"evidence": evidence, "label": label})
        except ValueError:
            # This catches 'variance' or any other non-numeric garbage
            continue

# separate data into training and testing groups 
evidence= [row["evidence" ] for row in data ]
labels = [row["label"] for row in data ]
X_training,X_testing,y_training,y_testing=train_test_split(
    evidence,labels,test_size=0.4
)

#create a neural network 
model=tf.keras.models.Sequential()  

# add a hidden layer with atleast 8 units 
model.add(tf.keras.layers.Dense(8,input_shape=(4,),activation="relu"))


# add input layer with one unit  with sigmoid activation
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

#train neural network

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
import numpy as np

# convert to numpy arrays
X_training = np.array(X_training, dtype=float)
X_testing = np.array(X_testing, dtype=float)
y_training = np.array(y_training, dtype=int)
y_testing = np.array(y_testing, dtype=int)

# train model
model.fit(X_training, y_training, epochs=20)

# evaluate
model.evaluate(X_testing, y_testing, verbose=2)

# train model
model.fit(X_training,y_training,epochs=20)
    
# evaluate how well 
model.evaluate(X_testing,y_testing, verbose=2)


