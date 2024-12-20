# **Lung Cancer Detection using Transfer Learning**

## Description

Now we will use the Kaggle API to download the dataset to the system. First, we will require the API key. To do this just navigate to the profile section in Kaggle and download the JSON file containing your details for API, after that just upload this to colab or locate in the local Jupyter environment.

## Table of Contents
- **Now we unzip the dataset into the desired folder**
- Dataset has been imported.
- **Data Visualization**
- These are the three classes that we have here.
- The above output may vary if you will run this in your notebook because the code has been implemented in such a way that it will show different images every time you re-run the code.
- **Data Preparation for Training**
- Some of the hyperparameters we can tweak from here for the whole notebook.
- One hot encoding will help us to train a model which can predict soft probabilities of an image being from each class with the highest probability for the class to which it really belongs.
- In this step, we will achieve the shuffling of the data automatically because train_test_split function split the data randomly in the given ratio.
- **Model Development**
- This is how deep this model. This also justifies why this model is highly affective in extracting useful features from images which helps us to build classifiers.
- 'Mixed7' is one of the layers in the inception network whose outputs we will use to build the classifier.
- While compiling a model we provide these three essential parameters:
- **Callback**
- Now we will train our model:
- Let's visualize the training and validation accuracy with each epoch.
- From the above graphs, we can certainly say that the model has not overfitted the training data as the difference between the training and validation accuracy is very low.
- **Model Evaluation**
- Let's draw the confusion metrics and classification report using the predicted labels and the true labels.
- Save the model

## How to Run

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```bash
   jupyter notebook Lung Cancer Detection using Transfer Learning.ipynb
   ```
3. Follow the steps in the notebook to preprocess data, train the model, and evaluate results.

## Dependencies

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Pandas

## Dataset

Ensure you have access to the dataset required for training and evaluation. Update the dataset path in the notebook if necessary.

## Results

The notebook demonstrates the implementation of a transfer learning model for lung cancer detection. The evaluation metrics and visualizations are generated at the end of the notebook.
