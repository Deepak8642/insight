import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def visualize_data(df):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    for column in numeric_columns:
        if df[column].nunique() > 10:
            # If the number of unique values is large, skip generating certain plots
            continue

        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, color='orange')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

        # Bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df.index, y=df[column])
        plt.title(f'Bar Plot of {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.show()

        # Scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df.index, y=df[column], data=df, color='green')
        plt.title(f'Scatter Plot of {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.show()

        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[[column]].corr(), annot=True, cmap='coolwarm')
        plt.title(f'Correlation Heatmap of {column}')
        plt.show()

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def main():
    file_path = 'preprocessed_data.csv'
    df = load_data(file_path)
    visualize_data(df)

if __name__ == "__main__":
    main()
