# nikhilburade-Titanic_Survival_Prediction

Titanic Survival Prediction Project Report
2. Methodology
TITANIC SURVIVAL PREDICTION USING MACHINE LEARNING
2.2 Data Preprocessing
The initial step in any machine learning project is cleaning and preprocessing the data. Several issues were
identified in the dataset:
1. Introduction
The Titanic disaster remains one of the most infamous shipwrecks in history. In this project, we use data
science techniques to analyze the Titanic dataset and build a machine learning model that predicts whether a
passenger survived or not. This project aims to demonstrate the complete lifecycle of a data science project -
from data preprocessing to visualization and model implementation - using Python.
- Missing Values: The 'Age' and 'Embarked' columns had missing entries. 'Age' was filled with the median,
while 'Embarked' was filled with the most frequent value.
2.1 Dataset Selection
The Titanic dataset provides an excellent starting point for beginners in data science and machine learning. It
is small, manageable, and contains both categorical and numerical features suitable for classification models.
The training dataset was used to build the model.
The dataset used in this study was obtained from Kaggle's "Titanic: Machine Learning from Disaster"
competition. It contains demographic and travel information about passengers on the Titanic, such as age,
Gender, ticket class, and whether they survived.
- Irrelevant Features: The 'Cabin', 'Name', and 'Ticket' columns were dropped as they had little impact on the
2.4 Visualization
To gain further insights, visualizations were created using Matplotlib and Seaborn. These included:
- Female passengers had a higher survival rate than male passengers.
- First-class passengers had significantly higher chances of survival compared to third-class passengers.
- Younger passengers, especially children, had slightly higher survival rates.
- Heatmap: Showed correlations between all numerical variables. The strongest correlation with survival was
with the 'Gender' and 'Pclass' features.
- Count Plot: Displayed the count of survivors and non-survivors, making the class imbalance clear.
- Histogram: Illustrated age distribution and survival probability across different age groups.
- Bar Chart: Compared survival rates across different passenger classes and embarkation points.
Correlation matrices and group-wise aggregations (e.g., survival rate by sex and class) were used to identify
the most influential features.
2.3 Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted to understand the relationships between features and the target
variable ('Survived'). Summary statistics such as mean, median, and standard deviation were computed. We
found several interesting patterns:
target variable or had too many unique values to encode effectively.
- Categorical Encoding: The 'Sex' column was encoded as binary (0 for male, 1 for female), and 'Embarked'
was one-hot encoded to convert categorical text data into numerical format.
- Duplicates: Duplicate entries were removed to ensure data consistency.
- These visualizations helped reinforce our statistical findings and added depth to our analysis.
- Features such as Pclass, Gender, Age, SibSp, Parch, Fare, and Embarked were used to train the model.
- The dataset was split into 80% training and 20% testing subsets.
- The model was trained using scikit-learn's LogisticRegression class and evaluated on the test data.
The model performed better on predicting non-survivors than survivors, likely due to the class imbalance in
the dataset. Further improvements could involve SMOTE for balancing, hyperparameter tuning, or using
ensemble models.

3.1 Logistic Regression
Logistic Regression was chosen as the base model due to its simplicity and interpretability. It is well-suited for
binary classification tasks like this one.
3.2 Evaluation
The model achieved an accuracy of approximately 80%, which is reasonable for a baseline model without
hyperparameter tuning. A classification report was generated to show precision, recall, F1-score, and support
for each class.
- Gender, Pclass, and Age were the most important features influencing survival.
- Visualizations confirmed the statistical findings: females and first-class passengers had better survival odds.
- The logistic regression model provided interpretable coefficients that matched our EDA insights.
This project serves as a strong foundation for beginners in data science. Future enhancements could include:
Overall, the Titanic dataset provided an accessible and insightful opportunity to learn the fundamental
concepts of machine learning and data analysis.
In this project, we successfully analyzed the Titanic dataset and built a basic machine learning model to
predict survival outcomes. The pipeline involved dataset selection, preprocessing, EDA, visualization, model
building, and evaluation - covering the entire lifecycle of a data science project.
