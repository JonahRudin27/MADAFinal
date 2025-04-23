import pandas as pd

def clean_data(file_path, n_rows_preview=5):
    # Load data
    df = pd.read_csv(file_path)

    # Step 1: Drop duplicates
    df = df.drop_duplicates()

    # Step 2: Handle missing values
    # Drop columns where more than 50% values are missing
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Fill remaining missing numeric values with column mean
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill remaining missing object (string) values with empty string
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("")

    # Step 3: Convert data types (optional)
    # Example: convert object columns that look like numbers to numeric
    for col in object_cols:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # Skip columns that can't be converted

    return df

def convert(df):
    std_df = pd.DataFrame()
    time_col = pd.to_datetime(
    df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )

    min_time = time_col.min()
    # Save months since min_time
    std_df['time'] = (time_col.dt.year - min_time.year) * 12 + (time_col.dt.month - min_time.month)

    age_map = {
        '0-4 years': 0, '5-11 years': 1, '12-17 years': 2, '18-29 years': 3,
        '30-39 years': 4, '40-49 years': 5, '50-64 years': 6, 
        '65-74 years': 7, '75 years and over': 8
    }

    std_df['age'] = df['subgroup2'].map(age_map)
    std_df['age'] = (std_df['age']-std_df['age'].mean())/std_df['age'].std()
    
    encoded = pd.get_dummies(df[['subgroup1', 'jurisdiction_residence']], drop_first=True)
    std_df = pd.concat([std_df, encoded], axis=1)
    normalized_rates = (df['crude_COVID_rate'] - df['crude_COVID_rate'].mean()) / df['crude_COVID_rate'].std()
    std_df['crude_COVID_rate'] = normalized_rates
    std_df = std_df.sort_values(by='time')
    return std_df


df = clean_data("covid_data.csv")
df = df[df["jurisdiction_residence"] != "United States"]
df = df[df["group"] == "Race and Age"]
df = df.groupby(['month', 'year', 'jurisdiction_residence', 'subgroup1', 'subgroup2'])['crude_COVID_rate'].sum().reset_index()
df = convert(df)


df.to_csv("df_encoded.csv", index=False)