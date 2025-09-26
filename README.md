Customer Churn Prediction with Deep Learning
============================================

Predict whether customers will churn using neural networks and structured customer data.

Project Summary
---------------

*   Input: features from Churn\_Modelling\_Dataset.csv
    
*   Architecture: feed-forward deep neural network
    
*   Task: binary classification of churn (0 = stay, 1 = churn)
    
*   Evaluated using accuracy, loss, confusion matrix
    

Motivation
----------

Churn is costly: retaining an existing customer is cheaper than acquiring a new one. This model aims to help businesses proactively identify customers at risk of leaving.

Repo Structure
--------------

FileDescriptionCustomer Churn Prediction.ipynbMain notebook with EDA, data preprocessing, model building, training, evaluationChurn\_Modelling\_Dataset.csvRaw dataset (customer attributes + churn label)

Workflow
--------

1.  **Data Loading & EDA**
    
    *   Read CSV, inspect distributions, missing values
        
    *   Visualize correlations and class imbalance
        
2.  **Preprocessing**
    
    *   Drop irrelevant columns (e.g. RowNumber, CustomerId, Surname)
        
    *   Encode categorical features (Geography, Gender)
        
    *   Scale numeric features (StandardScaler or equivalent)
        
    *   Split into train/test sets
        
3.  **Model Design & Training**
    
    *   Build a sequential neural network (Dense layers, activations, dropout, etc.)
        
    *   Compile with loss = binary\_crossentropy and optimizer (e.g. Adam)
        
    *   Train with monitoring (validation split or early stopping)
        
4.  **Evaluation**
    
    *   Compute test accuracy, test loss
        
    *   Plot training and validation curves (loss vs epochs)
        
    *   Generate confusion matrix, classification report
        
5.  **Inference / Prediction**
    
    *   Use trained model to make predictions on new data
        
    *   Interpret model outputs (probabilities → class)
        

Sample Results (replace with your real numbers)
-----------------------------------------------

*   Best architecture: 3 hidden layers with 64 → 32 → 16 units
    
*   Test accuracy: ~ **81.5 %**
    
*   Loss: ~ **0.45**
    
*   \[\[1532 98\] \[ 220 450\]\]
    
*   Notes: model converged in ~ 50 epochs, with early stopping at 45 epochs.
    

Key Skills Demonstrated
-----------------------

*   Deep learning using Keras / TensorFlow
    
*   Handling structured tabular data
    
*   Feature encoding and scaling
    
*   Model tuning (dropout, layer sizes)
    
*   Visualization of training dynamics
    
*   Model evaluation and confusion analysis
    

Setup & Running Instructions
----------------------------
```
git clone https://github.com/Rohit-Singh-0/Customer-Churn-Prediction-with-Deep-Learning-.git
cd Customer-Churn-Prediction-with-Deep-Learning-
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook "Customer Churn Prediction.ipynb"   `
```
If requirements.txt is missing, manually install:

``` 
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

Suggestions to Improve
----------------------

*   Add requirements.txt (or environment.yml)
    
*   Export the model (e.g. model.h5) and include a small inference script (predict.py)
    
*   Add a README\_badge for build status or metrics
    
*   Provide hyperparameter summary and experiments log
    
*   Wrap notebook logic into modular scripts for production / CI
    
*   Add comments, markdown explanations, and business insight section
