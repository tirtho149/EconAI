import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = {
    'Country': ['MMR', 'MOZ', 'SDN', 'GNB', 'SSD', 'COD', 'COG', 'NGA', 'SYR', 'LBN', 'SOM', 
                'IRQ', 'HTI', 'AFG', 'LAO', 'LBR', 'MLI', 'TCD', 'NER', 'CMR', 'CAF', 'YEM', 
                'GMB', 'BDI', 'BFA'],
    '1': [5, 5, 5, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 4],
    '2': [2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 3, 1, 2, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3],
    '3': [5, 5, 5, 5, 5, 4, 4, 5, 4, 4, 5, 4, 4, 4, 5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 4],
    '4': [4, 3, 2, 4, 2, 3, 3, 4, 3, 4, 2, 3, 3, 2, 3, 3, 2, 3, 2, 2, 3, 3, 3, 2, 4],
    '5': [4, 4, 4, 4, 4, 3, 3, 4, 5, 4, 3, 3, 4, 4, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4],
    '6': [5, 4, 4, 4, 4, 3, 4, 4, 5, 4, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4],
    'GroundTruth1': [6.06, 7.86, 6.81, 5.22, 2.47, 6.64, 1.24, 6.68, 1.62, 12.36,
                     1.25, 6.56, 41.49, 7.74, 3.58, 10.79, 7.64, 2.99, 8.21,
                     55.30, 4.91, 47.02, 34.15, 3.16, 13.22]
}

df = pd.DataFrame(data)

# Feature selection including columns 1 to 6
X = df[['1', '2', '3', '4', '5', '6']]

# Define a threshold for high inflation (for example, let's say 10)
threshold_value = 3
y = (df['GroundTruth1'] > threshold_value).astype(int)  # Create a binary target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
