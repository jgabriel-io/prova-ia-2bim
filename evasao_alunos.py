# ——————————————————————————————————————————————
# evasao_alunos_extensoes.py
# ——————————————————————————————————————————————

# 1. Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.tree        import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors   import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline    import Pipeline

from sklearn.metrics     import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 2. Carregando os dados
df = pd.read_csv('evasao_alunos.csv')
print("Dimensões:", df.shape)
print(df.head(), "\n")

# 3. EDA resumida
print(df.describe(), "\n")
print("Nulos por coluna:\n", df.isnull().sum(), "\n")

# Histogramas e boxplots
for col in ['frequencia', 'nota_media']:
    plt.figure(figsize=(6,4))
    plt.hist(df[df.evadiu==0][col], bins=10, alpha=0.6, label='Não evadiu')
    plt.hist(df[df.evadiu==1][col], bins=10, alpha=0.6, label='Evadiu')
    plt.title(f'{col} por classe'); plt.legend(); plt.show()
    plt.close()
    plt.figure(figsize=(4,4))
    plt.boxplot([df[df.evadiu==0][col], df[df.evadiu==1][col]],
                labels=['Não evadiu','Evadiu'])
    plt.title(f'Boxplot de {col}'); plt.show() 
    plt.close()


# 4. Pré-processamento e split
X = df[['frequencia','nota_media','bolsista']]
y = df['evadiu']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Definição de pipelines
# — Decision Tree (não precisa de scaler, mas colocamos pipeline para uniformidade)
pipe_dt = Pipeline([
    ('scaler', StandardScaler()),       # opcional pra DT
    ('dt', DecisionTreeClassifier(random_state=42))
])

# — Gaussian NB
pipe_nb = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

# — KNN (scaler fundamental)
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

pipelines = {
    'Decision Tree': pipe_dt,
    'Gaussian NB'  : pipe_nb,
    'KNN'          : pipe_knn
}

# 6. Cross-validation k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("=== Cross-validation (5-fold) sobre treino ===")
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# 7. GridSearchCV para hiperparametrização
grids = {
    'Decision Tree': {
        'dt__max_depth': [None, 3, 5, 7, 10],
        'dt__min_samples_split': [2, 5, 10]
    },
    'Gaussian NB': {
        'nb__var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    'KNN': {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform','distance']
    }
}

best_estimators = {}
print("\n=== GridSearchCV sobre treino ===")
for name, pipe in pipelines.items():
    grid = GridSearchCV(pipe, grids[name], cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_estimators[name] = grid.best_estimator_
    print(f"{name} -> best params: {grid.best_params_} | best acc: {grid.best_score_:.4f}")

# 8. Avaliação final no conjunto de teste
resumos = []
for name, model in best_estimators.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    # Impressão
    print(f"\n--- {name} (test set) ---")
    print(f"Acurácia: {acc:.4f}")
    print("Matriz de Confusão:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    resumos.append({
        'Modelo': name,
        'Acurácia': acc,
        'Prec(0)': cr['0']['precision'],
        'Rec(0)' : cr['0']['recall'],
        'F1(0)'  : cr['0']['f1-score'],
        'Prec(1)': cr['1']['precision'],
        'Rec(1)' : cr['1']['recall'],
        'F1(1)'  : cr['1']['f1-score'],
    })
    
    # Plot da matriz
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{name} — Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['Não evadiu','Evadiu'], rotation=45)
    plt.yticks(ticks, ['Não evadiu','Evadiu'])
    thresh = cm.max()/2
    for i,j in np.ndindex(cm.shape):
        plt.text(j,i,cm[i,j], ha='center',
                 color='white' if cm[i,j]>thresh else 'black')
    plt.ylabel('Verdadeiro'); plt.xlabel('Predito'); plt.tight_layout()
    plt.show()
    plt.close()


# 9. Tabela final de comparação
res_df = pd.DataFrame(resumos).round(4)
print("\n=== Comparativo Final ===")
print(res_df)

# 10. Feature importance (Decision Tree)
dt_model = best_estimators['Decision Tree'].named_steps['dt']
feat_imp = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances (Decision Tree):")
print(feat_imp, "\n")
plt.figure(figsize=(5,3))
feat_imp.plot(kind='bar')
plt.title('Importância das features — Decision Tree')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
plt.close()
