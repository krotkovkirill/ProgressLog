from pathlib import Path

import pandas as pd

data_path = Path(__file__).with_name("data.csv")
df = pd.read_csv(data_path)

print(df.head())

"""while True:
    number = 1
    print("---\033[1mСтажировка\033[0m---")
    print(f"*** Неделя {number} ***")
    message = input("Расскажите как прошла неделя стажировки:  ")
"""
