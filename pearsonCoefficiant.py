import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    'Volatility': [7.93, 12.03, 13.71, 13.74, 1.84, 6.99, 5.45, 7.15, 
                   11.77, 11.66, 3.64, 3.60, 18.17, 7.98, 8.26, 9.89, 
                   6.60, 9.54, 11.84, 24.77, 9.54, 22.75, 12.68, 12.58, 10.65]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate the correlation matrix
correlation_matrix = df[['1', '2', '3', '4', '5', '6', 'Volatility']].corr()

# Print the correlation matrix
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix between Questions and Inflation')
plt.show()
