import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tester import get_score

data_train = pd.read_csv('train.csv')
null_age = data_train['Age'].isnull()


def fill_age_nans(df):
    """
    Fills NaN age values with mean age of people with same title
    :param df: pandas DataFrame with NaNs in age column
    :return: pandas DataFrame with filled Age column
    """
    filled_df = df
    filled_df["Title"] = filled_df["Name"].str.extract(r'\b(Mr\.|Mrs\.|Miss\.|Master\.|Dr\.)\s', expand=False)
    filled_df['Age'] = filled_df['Age'].fillna(filled_df.groupby('Title')['Age'].transform('mean').round())
    return filled_df

def add_custom_columns(df):
    """
    Adds 'Companions' and 'IsAlone' columns to the DataFrame.
    Companions: int number of companions present
    IsAlone: bool value, True if Companions == 0
    :param df: pandas DataFrame
    :return:
    """
    df['Companions'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = df['Companions'] == 0
    return df

data_train_filled = add_custom_columns(fill_age_nans(data_train))

# Print options for DataFrames, useful when debugging, redundant in final script
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


clf = RandomForestClassifier()

def prepare_mydata(df):
    """
    Prepares data for model fitting, replacing all values with integers and dropping redundant data
    :param df: pandas DataFrame
    :return:
    """

    # Transform Sex from string to int
    sexes = sorted(df['Sex'].unique())
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)

    # Transform Embarked from string to dummy variables
    df = pd.concat(
        [df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val', dtype=int)], axis=1)

    df = df.drop(
        ['Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title', 'IsAlone', 'Embarked'], axis=1)
    return df



data_train_filled = prepare_mydata(data_train_filled).drop('PassengerId', axis=1)

train_data = data_train_filled.values

# Training data values, skip the first column 'Survived'
train_features = train_data[:, 1:]

# 'Survived' column values
train_target = train_data[:, 0]


# Fit the model to training data
clf = clf.fit(train_features, train_target)
clf_score = clf.score(train_features, train_target)
print(f"Mean accuracy of Random Forest: {clf_score:.4f}")


# Prepare data to predict
data_predict = pd.read_csv('test.csv')
data_predict = prepare_mydata(add_custom_columns(fill_age_nans(data_predict)))
data_pred_values = data_predict.values




# Test data values, skip the first column 'PassengerId'
test_x = data_pred_values[:, 1:]

# Predict the Survival values for the test data
test_y = clf.predict(test_x).astype(np.int64)

# Save the prediction to file
data_predict['Survived'] = test_y
data_predict[['PassengerId', 'Survived']] \
    .to_csv('titanic_prediction.csv', index=False)

get_score()