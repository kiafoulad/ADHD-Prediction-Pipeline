# Machine Learning Pipeline for Diagnosis Prediction

This project is a comprehensive, end-to-end machine learning pipeline designed to predict a mental illness diagnosis based on a dataset of ADHD-related features. The workflow demonstrates best practices in data science, from data cleaning and preprocessing to model evaluation and validation.

## Key Features

- **Robust Preprocessing Pipeline:** Utilizes Scikit-learn's `ColumnTransformer` to systematically handle mixed data types, including imputation, scaling for numerical data, and one-hot encoding for categorical data.
- **Imbalanced Data Handling:** Implements the SMOTE (Synthetic Minority Over-sampling Technique) to address significant class imbalance in the training data, preventing model bias.
- **Model Comparison:** Evaluates and compares the performance of multiple classifiers, including Random Forest, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).
- **Rigorous Validation:** Employs 5-fold cross-validation on the top-performing model to ensure its stability and reliability on unseen data.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- imblearn (for SMOTE)
- Matplotlib

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/diagnosis-prediction-pipeline.git](https://github.com/your-username/diagnosis-prediction-pipeline.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd diagnosis-prediction-pipeline
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the complete analysis, training, and evaluation pipeline, execute the main script:
```bash
python your_script_name.py
```
*(Note: Replace `your_script_name.py` with the actual name of your Python file.)*

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.