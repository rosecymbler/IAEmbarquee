![image](https://github.com/user-attachments/assets/3374207b-4a83-40dd-b63a-add09cf5f355)# Predictive Maintenance using AI4I 2020 Dataset

This project implements an end-to-end predictive maintenance solution for industrial equipment using machine learning. We developed a neural network model that processes sensor data to detect and classify five types of machine failures, with deployment capability on STM32 microcontrollers for edge computing applications.

## Dataset Overview

The dataset used is the **AI4I 2020 Predictive Maintenance Dataset**, which contains **10,000 instances** of industrial sensor data. Each instance represents the operating condition of a machine and is associated with a label indicating whether a failure has occurred and, if so, what type of failure it is.
**Key Challenge:** Extreme class imbalance (1939:1 ratio between normal operation and failure cases)

### Failure Types (Labels)

- **TWF**: Tool Wear Failure
- **HDF**: Heat Dissipation Failure
- **PWF**: Power Failure
- **OSF**: Overstrain Failure
- **RNF**: Random Failure


## Project Structure

The project is structured around a Jupyter notebook that guides through the following steps:

1. **Dataset Analysis**: Exploratory data analysis to understand the dataset. This analysis revealed a significant class imbalance, with functional machines being vastly overrepresented compared to failed ones. Such an imbalance could bias the model's learning, causing it to prioritize accuracy on the majority class while neglecting failure detection—a critical aspect for predictive maintenance. 
2. **Preparing the Dataset**: After cleaning the data and selecting relevant features (e.g., temperatures, speed, torque, wear, etc.), we split the dataset into training, test, and validation sets.
3. **Model Training without Balancing**: Training a deep learning model on the original, imbalanced dataset. When training a deep learning model on this dataset, the impact of class imbalance became evident in the model's performance, particularly when analyzing the confusion matrix.
4. **Model Training with Balancing**: To mitigate the issues observed in the unbalanced model training, we applied SMOTE (Synthetic Minority Oversampling Technique) from the imbalanced-learn library. This technique generates synthetic samples for the minority class (failed machines) to balance the dataset before training.
5. **Performance Comparison**: Comparing the performance of the models trained on imbalanced and balanced datasets. We observed improvements in the second confusion matrix, which implies, improvements in our model.

## STM32CubeIDE
In this final project stage, our goal was to deploy a neural network model - originally trained in a Jupyter Notebook - onto an STM32L4R9 microcontroller, enabling real-time local inference for predictive maintenance applications.

## Conclusion on the Embedded AI Model Performance

![WhatsApp Image 2025-03-24 at 17 54 21_fa6f9cbf](https://github.com/user-attachments/assets/ede1e31c-4f4b-4bf2-9a74-afdd37b0d3f8)


The results demonstrate the remarkable effectiveness of our AI model deployed on STM32CubeIDE. After a progressive learning phase, the system achieved **100% accuracy by the 100th iteration**, with consistent and stable convergence (improving from 0.91 to 0.99 between iterations 91 and 99).  

A particularly notable result occurred at iteration 98, where the model correctly identified a different pattern (**[0, 0, 1, 0, 0]**) with **99.6% confidence**, while maintaining high overall accuracy. The perfect match between expected and received outputs in the final iterations confirms the robustness of our embedded implementation on the STM32 platform.  

These excellent results validate our optimization approach for deployment on STM32 microcontrollers while ensuring reliable real-time inference for practical applications. The linear progression in accuracy suggests that the model could have converged even earlier with additional iterations.  

This successful implementation highlights the potential of lightweight AI models in resource-constrained embedded systems, balancing efficiency and performance.

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
- **STM32CubeIDE** : For testing our AI model

## Installation and Usage

1) Clone this repository
2) Ensure the necessary dependencies are installed
3) Open the TP_IA_EMBARQUEE.ipynb notebook with Jupyter Notebook or JupyterLab
4) Execute the notebook cells sequentially to follow the complete process
5) Download the file containing our AI model
6) Flash the model onto the STM32 board. Display the model's accuracy on the STM32 screen
7) Run inference on test data using the STM32 board. Confirm that the displayed accuracy matches expectations.
