# CreditIQ - Credit Score Predictor

CreditIQ is a Streamlit app that predicts whether a person’s credit score is **Good**, **Standard**, or **Poor** using a trained Random Forest model.

## Features

- Interactive Streamlit UI for credit score prediction
- Three app views: Predict, Model Info, and About
- Uses a preprocessing pipeline and label encoder saved with Joblib
- Displays predicted class, confidence, and model guidance
- Shows model metrics, best parameters, and the ML pipeline used during training

## Project Structure

- `app.py` - main Streamlit application
- `train.csv` - training dataset
- `test.csv` - test dataset
- `EDA_train_model.ipynb` - notebook used for exploration and model development
- `preprocessor.pkl` - fitted preprocessing pipeline
- `label_encoder.pkl` - fitted label encoder
- `credit_scoring_rf_model.pkl` - trained Random Forest model
- `requirements.txt` - Python dependencies

## Model Inputs

The app expects the following features:

- Age
- Occupation
- Annual_Income
- Monthly_Inhand_Salary
- Num_Bank_Accounts
- Num_Credit_Card
- Interest_Rate
- Num_of_Loan
- Delay_from_due_date
- Num_of_Delayed_Payment
- Changed_Credit_Limit
- Num_Credit_Inquiries
- Credit_Mix
- Outstanding_Debt
- Credit_Utilization_Ratio
- Payment_of_Min_Amount
- Total_EMI_per_month
- Amount_invested_monthly
- Payment_Behaviour
- Monthly_Balance
- Debt_to_Income
- EMI_to_Salary_Ratio
- Savings_Rate
- Delay_per_Loan
- Cards_per_Bank
- Credit_History_Months

Some of these are calculated automatically in the app from the values you enter.

## Requirements

Install the dependencies with:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

If you prefer, you can also add these packages to `requirements.txt`.

## How to Run

1. Open a terminal in the `Credit_scoring` folder.
2. Make sure these files are in the same directory as `app.py`:
   - `credit_scoring_rf_model.pkl`
   - `preprocessor.pkl`
   - `label_encoder.pkl`
3. Start the app:

```bash
streamlit run app.py
```

4. Open the local Streamlit URL shown in the terminal.

## Model Information

- Model type: Random Forest Classifier
- Classes: Good, Standard, Poor
- Reported accuracy: 79.05%
- Reported F1 score: 79.03%
- Trees: 200

## Notes

- The app uses a trained model artifact saved with Joblib.
- If the model file is missing, the app will stop and show a load error.
- `credit_scoring_rf_model.pkl` is not tracked in GitHub because it is larger than GitHub’s normal file size limit.
