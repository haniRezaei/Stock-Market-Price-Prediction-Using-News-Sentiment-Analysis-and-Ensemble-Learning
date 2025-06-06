# Stock-Market-Price-Prediction-Using-News-Sentiment-Analysis-and-Ensemble-Learning
This project implements a  NLP + Machine Learning model for predicting stock market movement (up/down) using financial news headlines. It is based on the IEEE 2022 paper:

Stock markets are influenced not only by numerical indicators but also by investor sentiment expressed in financial news. This project aims to Predict whether the DJIA index will go UP or DOWN based on daily news headlines.

News headlines from 2008 to 2016, Each row contains 25 headlines for a day, Labels: 1 if DJIA went up, 0 if it went down
#method:
1. Data Preparation: Combined 25 headlines into one text per day and then Preprocessed using tokenization, lowercasing, and stopword removal

2. Sentiment Analysis, Used VADER to extract compound, pos, neg, neu scores

3. Subjectivity Classification, Built a CNN-based classifier using GloVe embeddings and the Cornell Subjectivity dataset, Extracted SubjObj_Score (subjective/objective weight) for each day's news

4. Feature Engineering, VADER sentiment scores, SubjObj_Score, Historical market indicators (Open, High, Low, Close, Volume)

5. Modeling, Trained multiple classifiers: Random Forest, SVM, XGBoost

in this project we have also implement a deep learning model using Long Short-Term Memory (LSTM) networks to forecast future stock prices based purely on past historical closing prices of the Dow Jones Industrial Average (DJIA).


# Methodology
1. Preprocessing
Dates sorted chronologically, Converted to time series with proper datetime index, MinMax normalization applied to the Close column, 70/30 train-test split

2. Feature Engineering
Sliding window approach, Input (X): Last 60 days of closing prices, Output (Y): Closing price on day 61

3. LSTM Model Architecture, 2 stacked LSTM layers, Fully connected Dense layers, Mean Squared Error loss, Trained for 5 epochs

4. Evaluation Metrics, Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)




