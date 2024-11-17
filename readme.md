# SMS Spam Classification

This repository contains a machine learning model for classifying SMS messages as spam or not spam (ham). The project uses Python and several machine learning libraries, including scikit-learn, to preprocess the text data, train a model, and evaluate its performance.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)

- [Model Training](#model-training)

- [Contributing](#contributing)
- [License](#license)


## Overview
The goal of this project is to build a spam classifier using machine learning techniques. The model takes a text message as input and classifies it as either "spam" or "ham" (non-spam). The classifier is trained on a labeled dataset of SMS messages, with features extracted from the text (such as word counts, TF-IDF scores, etc.).

### Dataset
The dataset used in this project is a collection of SMS messages labeled as spam or ham. Each message is a short text, and it is either marked as spam or ham based on its content. The data can be found in the `data/` directory (or you can use your own labeled dataset).

## Prerequisites
Before running the code, ensure that you have the following installed:

- Python 3.x

Required Python libraries:
- `scikit-learn`
- `pandas`
- `numpy`
- `nltk`
- `matplotlib`
- `seaborn`

You can install the required dependencies by running the following command:

```bash
pip install scikit-learn pandas numpy nltk matplotlib seaborn

```

## Installation

Follow these steps to set up the project and start using it:

### 1. Clone the Repository

Clone this repository to your local machine using Git:

```bash
git clone https://github.com/SnehaVPujari007/sms-spam-classification.git

```
## Model Training

The model is trained using a text preprocessing pipeline and a classification algorithm. Below is an overview of the training process:

### 1. Data Preprocessing

- The dataset is cleaned by removing any non-alphabetic characters and converting all text to lowercase.
- The text data is tokenized, and stopwords are removed using NLTK's stopwords list.
- Features are extracted using the **TF-IDF vectorizer**, which converts the text data into a numerical representation.

### 2. Model Selection

The model is trained using a machine learning algorithm, such as:
- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machine (SVM)**

Hyperparameters are tuned for optimal performance using **cross-validation** to avoid overfitting and improve the model's generalization ability.

### 3. Saving the Model

After training, the model is saved to disk using either **joblib** or **pickle**. This allows the trained model to be reloaded and used for future predictions without retraining.

```python
import joblib

# Saving the model
joblib.dump(model, 'spam_classifier_model.pkl')

```
## Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow these steps:

1. **Fork** the repository to your own GitHub account.
2. **Create a new branch** for your feature or bug fix.
3. **Make the necessary changes** and commit them with clear, descriptive commit messages.
4. **Push your changes** to your forked repository.
5. **Submit a pull request** with a detailed description of the changes you’ve made.

Before contributing, please ensure your code adheres to the existing coding standards and includes appropriate tests. If you encounter any issues or have suggestions for improvements, feel free to open an issue in the repository.

## License

This project is licensed under the **MIT License**. Feel free to use and modify the code for your own projects.
