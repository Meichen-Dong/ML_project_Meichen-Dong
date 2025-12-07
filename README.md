# Diamond Price Prediction — Machine Learning Project

This project aims to build a supervised machine learning model to predict the price of a diamond based on its physical and quality characteristics.
The analysis including data preprocessing(demonstration and cleaning), model training(Random forest, Linear regression, XGBoost), hyperparameter optimization, and model interpretation.

Finally, a streamlit web application is created for interactive prediction (The results based on the XGBoost model, which performs best -- lowest RMSE and the R^2 is closest to 1)

## 1. Project Goal
The goal of this project is to use the real-world dataset from Kaggle “Diamonds Dataset” to adress the predictive question: “Can we predict the price of a diamond from its physical and quality attributes?”

## 2. Dataset Description
Source: Kaggle — Diamonds Dataset (https://www.kaggle.com/datasets/ritikmaheshwari/diamond-price-prediction/data?select=Diamond_Price_Prediction.ipynb)

Dataset including 53940 samples and 10 features originally.

Target variable: price (continuous) -- In the model, we use its logrithm value, which can helps handle the variance

Feature	Description:
carat	    (continuous) Weight of the diamond
depth	    (continuous) Depth percentage = height / avg diameter * 100
table	    (continuous) The ratio of the width of the largest flat surface at the top of the diamond (called the 
                        "table facet") to the widest part of the diamond.
x, y, z	    (continuous) Diamond dimensions in Width, Length, Height
cut	        (categorical) The cut type of the Diamond, it determines the shine (Fair, Good, Very Good, Premium, 
                          Ideal)
color	    (categorical) Graded from D (best) to J (worst)
clarity	    (categorical) Graded from I1 (worst) to IF (best)
price	    (continuous) Target variable

Additional features:
volume      (continuous) x * y * z
density     (continuous) carat / volume
xy_ratio    (continuous) x / y (This value measures whether a diamond is close to a perfect circle. The closer the 
                         value to 1, the closer the diamond is a perfect circle)

I add these engineered features to improve the model performance.

## 3. Preprocessing 
For variables deleting or splitting :
My dataset donnot contain missing values, and date variable. But in the beginning, I delete 10 samples with a value 0 in Width, Length or Height, cause it will result in some errors when compute additional variable Density (Carat /(x * y * z)) or xy_ratio (x/y)

For categorical variable encoding:
In the beginning, I try to use pd.Categorical(col).codes to process all the categorical varibles. However, I realized that this method typically assigns numbers alphabetically or in the order of first appearance, and it cannot link the meaning of categorical variable values ​​to the values ​​of their corresponding numerical variables. And I only have 3 categorical varibles, so I define their numeric variables by myself. (cut: Fair → Ideal (1–5), color: J → D (1–7), clarity: I1 → IF (1–8))

## 4. Model Training & Optimization
In the project, I test 3 regression models: Linear Regression, Random Forest and XGBoost(best performing)

For Random Forest and XGBoost, I use the optuna methed to optimize.

Finally, the best-performing model is XGBoost, trained on logarithmic price for more stable prediction.

## 5. Model Performance
Using the XGBoost model:
Metric	Training Set	Validation Set
RMSE	   0.051	        0.081
R²	       0.997	        0.993

Very small generalization gap → no overfitting
High R² → model explains ~99% of price variance
Low RMSE → high precision in price prediction

## 6. Model Interpretation
To interpret the xgboost model, I used permutation importance method, which measures how much the model’s performance decreases when each feature is randomly shuffled.

The results show:
Carat is the most important feature (around 0.42), meaning diamond carat affect its value most.
y (width) is the second most influential geometric feature (around 0.34).
Clarity and color also contribute meaningfully, but other features such as x, z, volume, xy_ratio, depth, table, density, and cut have smaller impact and play more supporting roles.

## 7. Streamlit Application
1. Open the website (run "streamlit run app.py" in Terminal or just click on the link https://mlprojectmeichen-dong-nhieqvaesfxssg8u3mpp4k.streamlit.app/ ).
2. Use the sidebar on the left to input the diamond’s characteristics as you want, then the app will automatically computes the engineered features (volume, density, xy_ratio).
3. Click “Click me to predict” to generate the price estimate.

