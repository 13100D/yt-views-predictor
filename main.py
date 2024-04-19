from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

data = pd.read_csv("IN_youtube_trending_data.csv")
drop = ['title', 'channelTitle', 'tags', 'thumbnail_link', 'description']
for i in drop:
    data.pop(i)
data_cleaned = data[(data['comments_disabled'] & data['ratings_disabled'])]
data_cleaned.pop('comments_disabled')
data_cleaned.pop('ratings_disabled')

data_cleaned['interaction'] = data['likes'] + data['dislikes'] + data['comment_count']

data_cleaned['publishedAt'] = pd.to_datetime(data_cleaned['publishedAt'])
data_cleaned['trending_date'] = pd.to_datetime(data_cleaned['trending_date'])
data_cleaned['trendTime'] = (data_cleaned['trending_date'] - data_cleaned['publishedAt']).dt.total_seconds() // 3600

data_cleaned['publishedAt'] = pd.to_datetime(data_cleaned['publishedAt'], utc=True)
current_datetime = datetime.now(timezone.utc)
data_cleaned['publishedTime'] = (current_datetime - data_cleaned['publishedAt']).dt.total_seconds() // 3600

print(data_cleaned.dtypes)

X = data_cleaned[['publishedTime', 'trendTime', 'interaction']]
y = data_cleaned['view_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)

for i in range(1,100):
    regressor = DecisionTreeRegressor(max_depth=i, criterion='squared_error', random_state=69)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    r2 = r2_score(y_test, y_pred)
    print("R-squared (R^2):", r2)