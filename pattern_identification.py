import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# Function to define a simple neural network model for classification using TensorFlow
def build_classification_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Function to define a simple neural network model for regression using PyTorch
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def identify_patterns(df):
    # Exclude non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Clustering using K-Means
    kmeans = KMeans(n_clusters=3)
    df['KMeans_Cluster'] = kmeans.fit_predict(df[numeric_columns])

    # Clustering using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(df[numeric_columns])

    # Visualize clusters
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numeric_columns])
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['KMeans_Cluster'])
    plt.title('K-Means Clustering')
    plt.show()

    # Classification using Logistic Regression
    if 'target_column' in df.columns:
        X = df.drop('target_column', axis=1)
        y = df['target_column']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)

        print("Classification Report for Logistic Regression:")
        print(classification_report(y_test, y_pred))

    # Regression using Linear Regression
    if 'target_column_regression' in df.columns:
        X_reg = df.drop('target_column_regression', axis=1)
        y_reg = df['target_column_regression']
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

        lin_reg = LinearRegression()
        lin_reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = lin_reg.predict(X_test_reg)

        mse = mean_squared_error(y_test_reg, y_pred_reg)
        print("Mean Squared Error for Linear Regression:", mse)

    # Classification using TensorFlow neural network
    if 'target_column_nn' in df.columns:
        X_nn = df.drop('target_column_nn', axis=1)
        y_nn = df['target_column_nn']
        X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.3, random_state=42)

        input_shape = (X_train_nn.shape[1],)
        num_classes = len(np.unique(y_nn))

        model = build_classification_model(input_shape, num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_nn, y_train_nn, epochs=10, batch_size=32, validation_data=(X_test_nn, y_test_nn))

    # Regression using PyTorch neural network
    if 'target_column_nn_reg' in df.columns:
        X_nn_reg = df.drop('target_column_nn_reg', axis=1).values
        y_nn_reg = df['target_column_nn_reg'].values
        X_train_nn_reg, X_test_nn_reg, y_train_nn_reg, y_test_nn_reg = train_test_split(X_nn_reg, y_nn_reg, test_size=0.3, random_state=42)

        input_size = X_train_nn_reg.shape[1]
        model = RegressionModel(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(100):
            inputs = torch.tensor(X_train_nn_reg, dtype=torch.float32)
            labels = torch.tensor(y_train_nn_reg, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            inputs = torch.tensor(X_test_nn_reg, dtype=torch.float32)
            labels = torch.tensor(y_test_nn_reg, dtype=torch.float32)
            outputs = model(inputs)
            mse_nn_reg = mean_squared_error(labels.numpy(), outputs.numpy())
            print("Mean Squared Error for PyTorch Neural Network Regression:", mse_nn_reg)

def main():
    file_path = 'preprocessed_data.csv'
    df = load_data(file_path)
    identify_patterns(df)

if __name__ == "__main__":
    main()