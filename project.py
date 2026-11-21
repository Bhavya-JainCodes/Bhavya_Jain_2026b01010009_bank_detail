import pandas as pd
import numpy as np


df = pd.read_csv('bank.csv')
print("Original Shape:", df.shape)

# 2. Duplicate rows remove karna
df = df.drop_duplicates().reset_index(drop=True)

# 3. Text columns ko clean aur standardize karna (lowercase + extra spaces hatana)
text_cols = df.select_dtypes(include='object').columns
for col in text_cols:
    df[col] = df[col].str.strip().str.lower()

# 'job' column mein 'admin.' se extra dot (.) hatana
df['job'] = df['job'].str.replace('.', '', regex=False)

# 4. 'unknown' values ko handle karna
# Pehle 'unknown' ko NaN (missing value) mein convert karein
df.replace('unknown', np.nan, inplace=True)

# 'job' aur 'education' mein missing values ko Mode (sabse zyada aane wali value) se fill karein
df['job'] = df['job'].fillna(df['job'].mode()[0])
df['education'] = df['education'].fillna(df['education'].mode()[0])

# 'contact' aur 'poutcome' mein 'unknown' zyada hai, isliye 'not_specified' label dein
df['contact'] = df['contact'].fillna('not_specified')
df['poutcome'] = df['poutcome'].fillna('not_specified')

# 5. 'pdays' mein -1 ko handle karna (-1 ka matlab pehle contact nahi hua)
df['previously_contacted'] = np.where(df['pdays'] == -1, 0, 1)
df['pdays'] = np.where(df['pdays'] == -1, 0, df['pdays'])

# 6. Outliers handle karna (IQR Method + NumPy clip)
num_cols = ['age', 'balance', 'duration', 'campaign']
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# 7. 'yes'/'no' columns ko Binary (1/0) mein convert karna
binary_cols = ['default', 'housing', 'loan', 'deposit']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# 8. Final Verification aur Clean Data save karna
print("Cleaned Shape:", df.shape)
print("Total Missing Values:", df.isnull().sum().sum())
print(df.head())

df.to_csv('bank_cleaned.csv', index=False)
print("File 'bank_cleaned.csv' successfully save ho gayi hai!")