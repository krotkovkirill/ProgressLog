from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

import pandas as pd

data_path = Path(__file__).with_name("data.csv")
df = pd.read_csv(data_path)

print(df.head())

vectorizer = TfidfVectorizer()

X = df["text"]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = Ridge()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for actual, predicted in zip(y_test, y_pred):
    error = abs(actual - predicted)
    print(f"Твоя оценка: {actual} | Прогноз: {predicted:.2f} | Ошибка: {error:.2f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Средняя абсолютная ошибка: {mae:.2f}")

message = input("Введите сообщение для оценки: ")

features = vectorizer.transform([message])
score = model.predict(features)[0]
print(f'Оценка дня: {score:.2f}')










"""while True:
    number = 1
    print("---\033[1mСтажировка\033[0m---")
    print(f"*** Неделя {number} ***")
    message = input("Расскажите как прошла неделя стажировки:  ")
"""
