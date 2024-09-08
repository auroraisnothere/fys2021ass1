


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def loading_data():

    # Read the file

    songs = pd.read_csv('SpotifyFeatures.csv')

    classes = songs[songs["genre"].isin(["Pop", "Classical"])]

    number_of_songs = songs.shape[0] 
    number_of_features = songs.shape[1]

    # This method "map" applies and returns the value for each class

    classes["genre"] = songs["genre"].map({"Pop": 1, "Classical": 0})

    # Classifying the genres based on the features. The .values function returns the data as a numpy array

    features = classes[["liveness", "loudness"]].values
    labels = classes[["genre"]].values

    # How many samples belong to each class

    counts = classes["genre"].value_counts()

    # creates a vector of ones with the same number of rows as the features matrix. 
    # This is bias column for further logistic regression

    n_samples = features.shape[0]
    ones_column = np.ones((n_samples, 1))
    X = np.concatenate([ones_column, features], axis=1) # .concatenate

    return X, labels, counts, number_of_songs, number_of_features

X, y,counts, number_of_songs, number_of_features = loading_data()


print(f"number of songs: {number_of_songs}")
print(f"number of features: {number_of_features}")

# 1 is the label which should be counted and 0 is what should be returned if there are no label 1 found
print(f"number of pop songs: {counts.get(1, 0)}") 
print(f"number of classical songs: {counts.get(0, 0)}")


# TRAINING SET

# Separating the training data 

samples = len(y) # The length of the y vector
num_train_samples = int(samples * 0.8) # Calculating 80% of the data
X_train = X[:num_train_samples] # training data for the input features
y_train = y[:num_train_samples] # training data for the target labels
X_test = X[num_train_samples:]  # test data for the input features
y_test = y[num_train_samples:]  # test data for target labels



# LOGISTIC DISCRIMINATION CLASSIFIER

def sigmoid(z):
    return 1/(1 + np.e**(-z))       

def logistic_regression(X_train, y_train, epochs, learning_rate):

    loss_data = [] # An empty list where the loss value for each iteration will be stored
    beta = np.zeros(X_train.shape[1]) # Parameter for logistic regression

    for epoch in range(epochs):

        indices = np.random.permutation(len(y_train)) # randomizes the data
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]


        # STOCHASTIC GRADIENT DESCENT
        for i in indices:

            X_i = X_train_shuffled[i] # The feature vector
            y_i = y_train_shuffled[i] # The true label

            z = np.dot(X_i, beta)
            y_pred = sigmoid(z)

            # Compute the gradient and update the parameter
            error = y_i.flatten() - y_pred
            gradient = error * y_pred * (1 - y_pred) * X_i
            beta += learning_rate * gradient

        # Compute loss for the entire dataset
        z_train = np.dot(X_train, beta)
        y_train_pred = sigmoid(z_train)

        # Binary cross entropy loss
        loss = -np.mean(y_train.flatten() * np.log(y_train_pred + 1e-8) + (1 - y_train.flatten()) * np.log(1 - y_train_pred + 1e-8))
        loss_data.append(loss)

        #if (epoch + 1) % 1000 == 0: # To avoid printing out too many values
            #print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

    
    return beta, loss_data, y_train_pred

epochs = 1000
learning_rate = 0.01
batch_size = 32

beta, loss_data, y_train_pred = logistic_regression(X_train, y_train, epochs, learning_rate)

# Calculate accuracy train set

#z_train = np.dot(X_train, beta) 
#y_train_pred = 1 / (1 + np.exp(-z_train)) 

# The probabilities should have a threshold of 0.5
y_train_pred_label = (y_train_pred >= 0.5).astype(int)

# Compare the predicted labels with the true labels.
# Flatten is used to make sure that the comparison is correct
accuracy_train = np.mean(y_train_pred_label.flatten() == y_train.flatten())
print(f'Accuracy on the training set: {accuracy_train * 100:.2f}%') # Calculate the percentage
    
# Plot training error as a function of epochs
plt.plot(loss_data, label=f'LR={learning_rate}')
plt.xlabel('Epochs')
plt.ylabel('Training Error (Loss)')
plt.legend()
plt.show()


# TEST SET
# Calculate accuracy test set

# Sigmoid function for test data
z_test = np.dot(X_test, beta)
y_test_pred = 1 / (1 + np.exp(-z_test))  

# convert the tests predicted labels to binary numbers with a threshold of 0.5 
y_test_pred_label = (y_test_pred >= 0.5).astype(int)

accuracy_test = np.mean(y_test_pred_label.flatten() == y_test.flatten())
print(f'Accuracy on the test set: {accuracy_test * 100:.2f}%')

# CONFUSION MATRIX

cm = confusion_matrix(y_test.flatten(), y_test_pred_label.flatten())

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["Classical", "Pop"],
            yticklabels=["Classical", "Pop"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



print("Confusion Matrix:")
print(cm)




