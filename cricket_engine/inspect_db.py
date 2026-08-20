import sqlite3
import pandas as pd

# Connect to your database
conn = sqlite3.connect("predictions.db")

# Load all rows and columns into a Pandas DataFrame
df = pd.read_sql_query("SELECT * FROM predictions", conn)
conn.close()

# Set pandas display options to see all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print(f"Total rows in database: {len(df)}")
print(df.head(10))  # Print first 10 rows