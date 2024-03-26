import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Page configurations
st.set_page_config(page_title="TensorFlow Playground", page_icon=":robot:")

# Function to plot decision surface using mlxtend
def plot_decision_surface(X_train, y_train, X_test, y_test, model):
    h = 0.02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) > 0.5
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    plot_decision_regions(X=X_train, y=y_train.astype(np.int_), clf=model, ax=ax, legend=2)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    return fig

# Sidebar
page = st.sidebar.radio("Go to", ["Page 1", "Page 2"])

if page == "Page 1":
    st.subheader("TensorFlow Playground of a Random Dataset")

    # Parameters
    st.sidebar.markdown("### Parameters")
    num_datapoints = st.sidebar.slider("Number of Datapoints", min_value=50, max_value=1000, value=100)
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1])
    noise = st.sidebar.slider("Noise", min_value=0.0, max_value=1.0, value=0.1)
    train_percentage = st.sidebar.slider("Train Percentage", min_value=0, max_value=100, value=70)
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=100, value=32)
    activation_function = st.sidebar.selectbox("Activation Function", ["sigmoid", "tanh", "relu", "linear"])

    # Generate random data
    np.random.seed(0)
    X = np.random.rand(num_datapoints, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Add noise
    X += noise * np.random.randn(num_datapoints, 2)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_percentage) / 100, random_state=42)

    # Create Sequential model
    model = Sequential()
    model.add(Dense(32, input_dim=2, activation=activation_function))
    model.add(Dense(16, activation=activation_function))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

    # Plot raw data
    st.subheader("Raw Random Data")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    st.pyplot(fig)

    # Evaluate the model
    _, train_score = model.evaluate(X_train, y_train, verbose=0)
    _, test_score = model.evaluate(X_test, y_test, verbose=0)

    # Display train/test score
    st.subheader("Train/Test Score")
    st.write(f"Train Score: {train_score}")
    st.write(f"Test Score: {test_score}")

    # Plot decision surface
    st.subheader("Decision Surface")
    fig = plot_decision_surface(X_train, y_train, X_test, y_test, model)
    st.pyplot(fig)

    # Plot train/test loss graph
    st.subheader("Train vs. Test Loss Graph")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Test Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Train vs. Test Loss')
    ax.legend()
    st.pyplot(fig)

elif page == "Page 2":
    st.subheader("Non-linear Dataset")

    # Load non-linear datasets
    datasets = {
        "ushape": r"C:\Users\HP\Desktop\Tensorflow playground-renuka\1.ushape.csv",
        "concentric_circle_1": r"C:\Users\HP\Desktop\Tensorflow playground-renuka\2.concerticcir1.csv",
        "concetric_circle_2": r"C:\Users\HP\Desktop\Tensorflow playground-renuka\3.concertriccir2.csv",
        "linear_sep": r"C:\Users\HP\Desktop\Tensorflow playground-renuka\4.linearsep.csv",
        "outlier": r"C:\Users\HP\Desktop\Tensorflow playground-renuka\5.outlier.csv",
        "overlap": r"C:\Users\HP\Desktop\Tensorflow playground-renuka\6.overlap.csv",
        "xor": r"C:\Users\HP\Desktop\Tensorflow playground-renuka\7.xor.csv",
        "two_spirals": r"C:\Users\HP\Desktop\Tensorflow playground-renuka\8.twospirals.csv"
    }

    dataset_name = st.sidebar.selectbox("Select Dataset", list(datasets.keys()))
    dataset_path = datasets[dataset_name]
    dataset = pd.read_csv(dataset_path)

    # Parameters
    st.sidebar.markdown("### Parameters")
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1])
    noise = st.sidebar.slider("Noise", min_value=0.0, max_value=1.0, value=0.1)
    train_percentage = st.sidebar.slider("Train Percentage", min_value=0, max_value=100, value=70)
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=100, value=32)
    activation_function = st.sidebar.selectbox("Activation Function", ["sigmoid", "tanh", "relu", "linear"])

    # Process dataset
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Add noise
    X += noise * np.random.randn(X.shape[0], X.shape[1])

    # Split dataset
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=(100 - train_percentage) / 100, random_state=42)

    # Create Sequential model
    model = Sequential()
    model.add(Dense(32, input_dim=2, activation=activation_function))
    model.add(Dense(16, activation=activation_function))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

    # Plot raw data
    st.subheader("Raw Data")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    st.pyplot(fig)

    # Evaluate the model
    _, train_score = model.evaluate(X_train, y_train, verbose=0)
    _, test_score = model.evaluate(X_test, y_test, verbose=0)

    # Display train/test score
    st.subheader("Train/Test Score")
    st.write(f"Train Score: {train_score}")
    st.write(f"Test Score: {test_score}")

    # Plot decision surface
    st.subheader("Decision Surface")
    fig, ax = plt.subplots()
    plot_decision_regions(X=X_train, y=y_train.astype(np.int_), clf=model, legend=2)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    st.pyplot(fig)

    # Plot train/test loss graph
    st.subheader("Train vs. Test Loss Graph")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Test Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Train vs. Test Loss')
    ax.legend()
    st.pyplot(fig)


