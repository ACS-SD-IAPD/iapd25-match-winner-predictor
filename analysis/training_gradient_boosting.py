import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt


from global_vars import BASE_DIR


df = pd.read_csv(BASE_DIR + '/data/team_form_final.csv')
X = df.drop(['rezultat'], axis=1)
y = df['rezultat']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Acuratete: {cv_scores.mean():.4f}')
print(classification_report(y_test, y_pred, target_names=['Egal', 'Gazda', 'Oaspete']))

fi = pd.DataFrame({
    'Feature': X.columns,
    'Importanta': model.feature_importances_
}).sort_values('Importanta', ascending=False)

print("\nTop feature-uri")
print(fi)
cm = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, xticklabels=['Egal', 'Gazda', 'Oaspete'], yticklabels=['Egal', 'Gazda', 'Oaspete'], cmap='Oranges')
plt.xlabel('Predictie')
plt.ylabel('Realitate')
plt.title('Matricea de Confuzie')
plt.show()
