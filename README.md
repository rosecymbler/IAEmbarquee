# Predictive Maintenance using AI4I 2020 Dataset

This project focuses on predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset. The goal is to create a deep learning model that can predict machine failures based on sensor data.

## Dataset Overview

The dataset used is the **AI4I 2020 Predictive Maintenance Dataset**, which contains **10,000 instances** of industrial sensor data. Each instance represents the operating condition of a machine and is associated with a label indicating whether a failure has occurred and, if so, what type of failure it is.

### Failure Types (Labels)

- **TWF**: Tool Wear Failure
- **HDF**: Heat Dissipation Failure
- **PWF**: Power Failure
- **OSF**: Overstrain Failure
- **RNF**: Random Failure


## Project Structure

The project is structured around a Jupyter notebook that guides through the following steps:

1. **Dataset Analysis**: Exploratory data analysis to understand the dataset.
2. **Model Training without Balancing**: Training a deep learning model on the original, imbalanced dataset.
3. **Model Training with Balancing**: Training a deep learning model on a balanced dataset using techniques from the `imbalanced-learn` library.
4. **Performance Comparison**: Comparing the performance of the models trained on imbalanced and balanced datasets.

## Authors

- **Rose Cymbler**
- **Ness Tchenio**

## Technologies Used

- **Python**
- **TensorFlow/Keras**: For building and training the deep learning model.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For additional machine learning utilities.
- **Numpy**: For numerical operations.
- **Imbalanced-learn**: For balancing the dataset.

## Installation and Usage

1) Clone this repository
2) Ensure the necessary dependencies are installed
3) Open the TP_IA_EMBARQUEE.ipynb notebook with Jupyter Notebook or JupyterLab
4) Execute the notebook cells sequentially to follow the complete process
5) Download the file containing our AI model
6) Use it in a new STM32 project, usin
