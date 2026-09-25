import numpy as np
import pandas as pd
data =r'C:\Users\Admin\Desktop\ml\pd.csv.xlsx'
df = pd.read_excel(data)
print("Data Frame") 
print(df)
print("\n Retrived column (marks):")
print(df["Marks"])
print("\n Sum of marks:")
print((np.sum(df["Marks"])))
print("\nMean of all numeric columns:")
print(df.mean(numeric_only=True))
print("\n Standard Deviation:")
print(df.std(numeric_only=True))
