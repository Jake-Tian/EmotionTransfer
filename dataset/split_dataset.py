# split the dataset into train and test
import pandas as pd

df = pd.read_csv('yelp_labeled.csv')

# split the dataset into train and test
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# save the train and test datasets
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)