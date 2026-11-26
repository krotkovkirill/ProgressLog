from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

import pandas as pd

data_path = Path(__file__).with_name("data.csv")
df = pd.read_csv(data_path)

vectorizer = TfidfVectorizer()

X = df["text"]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = Ridge()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

"""
for actual, predicted in zip(y_test, y_pred):
    error = abs(actual - predicted)
    print(f"Твоя оценка: {actual} | Прогноз: {predicted:.2f} | Ошибка: {error:.2f}")
"""

mae = mean_absolute_error(y_test, y_pred)
print(f"Средняя абсолютная ошибка: {mae:.2f}\n")

scores = []
week = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

for day in range(1, 8):
    message = input(f'введите как прошел ваш {week[day-1]}: ')
    if message.strip() == 'f':
        scores.append(None)
        continue
    features = vectorizer.transform([message])
    score = model.predict(features)[0]
    scores.append(score)

for day, score in zip(range(1, 8), scores):
    if score is None:
        print(f"Прогноз на {week[day-1]}: день гавно")
    else:
        print(f"Прогноз на {week[day-1]}: {score:.1f}")


rate_scores = [score for score in scores if score is not None]
plot_scores = [float('nan') if score is None else score for score in scores]

print(f"Средняя оценка за неделю: {sum(rate_scores) / len(rate_scores):.1f}")

plt.plot(week, plot_scores, marker='o')
plt.title('Прогноз на неделю')
plt.ylim(0, 10)
plt.show()

