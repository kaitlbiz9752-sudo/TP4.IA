## Ce travaille réalisée par AITLBIZ Kaoutar 
## Etudiante MASTER MDSIE
## ENS Marrakech



--------



## TP — Apprentissage non supervisé : Clustering avec K-Means
**Objectifs pédagogiques**

À la fin de ce TP, vous serez capable de :

Charger et explorer un jeu de données non étiqueté.

Prétraiter et normaliser les données.

Appliquer l’algorithme K-Means pour créer des clusters.

Évaluer la qualité du regroupement.

Visualiser et interpréter les résultats.



-------------





## Étape 1 — Chargement des données
**Objectif**

Importer le jeu de données et observer sa structure.

**Dataset utilisé**

**Mall Customers Dataset**

Ce dataset contient des informations sur des clients de centre commercial : âge, revenu annuel et score de dépenses.


| Colonne                    | Description                                         |
| :------------------------- | :-------------------------------------------------- |
| **CustomerID**             | Identifiant unique du client                        |
| **Gender**                 | Sexe du client                                      |
| **Age**                    | Âge du client                                       |
| **Annual Income (k$)**     | Revenu annuel (en milliers de dollars)              |
| **Spending Score (1-100)** | Score de dépenses attribué par le centre commercial |



**Exemple de code**

```python
import pandas as pd
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
print(df.shape)
```



**AFFICHAGE**


<img width="786" height="198" alt="image" src="https://github.com/user-attachments/assets/281dad75-4576-4c3c-b1ab-198294845b19" />




**Interprétation**

Cette étape permet de comprendre les variables et leur type (numérique, catégoriel).




----------------









## Étape 2 — Prétraitement des données
**Objectif**

Nettoyer et préparer les données avant le clustering.

**Étapes**

Vérifier la présence de valeurs manquantes.

Sélectionner les variables numériques pertinentes (Age, Annual Income, Spending Score).

Appliquer une normalisation ou standardisation pour uniformiser les échelles.

**Exemple de code**

```python
from sklearn.preprocessing import StandardScaler

print(df.isnull().sum())  # Vérification des valeurs manquantes

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
```

**Résultat attendu**

Les variables sont centrées et réduites, prêtes pour K-Means.


**AFFICHAGE**

<img width="381" height="199" alt="image" src="https://github.com/user-attachments/assets/8e7ac701-a15e-471d-94a2-8eb15e4e4ee8" />






---------------






## Étape 3 — Visualisation exploratoire
**Objectif**

Observer visuellement d’éventuels regroupements dans les données.

**Exemple de code**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.title("Relation entre le revenu annuel et le score de dépenses")
plt.show()
```


**AFFICHAGE**



<img width="830" height="601" alt="image" src="https://github.com/user-attachments/assets/f6d3846d-a9bc-4b3b-840d-4711e3b66e9e" />




**Interprétation**

Les points semblent se regrouper naturellement selon les revenus et le score de dépenses.






-------------









## Étape 4 — Application de K-Means
**Objectif**

Créer des clusters avec l’algorithme K-Means.

**Exemple de code**

```python
from sklearn.cluster import KMeans

k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

print("Centres des clusters :\n", kmeans.cluster_centers_)
```

**Résultat attendu**

Chaque client est associé à un cluster, et les centres indiquent les moyennes des caractéristiques.



**AFFICHAGE**


<img width="498" height="150" alt="image" src="https://github.com/user-attachments/assets/796c8167-5fcb-46ab-9e36-d7d82556d296" />





-----------------







## Étape 5 — Évaluation du clustering
**Objectif**

Évaluer la cohérence des clusters.

**Exemple de code**

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, labels)
print("Silhouette Score :", score)
```






**Interprétation**

Score proche de 1 → très bons clusters.

Score proche de 0 → clusters qui se chevauchent.

Score négatif → mauvais regroupement.










--------------










## Étape 6 — Visualisation des clusters
**Exemple de code**



```python
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:,1], y=X_scaled[:,2], hue=labels, palette='viridis')
plt.title(f"K-Means (k={k})")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()
```



**AFFICHAGE**




<img width="954" height="705" alt="image" src="https://github.com/user-attachments/assets/c9502953-279d-4857-b02e-4877ddc26a79" />




**Résultat attendu**

Chaque couleur correspond à un cluster différent (ex. clients à haut revenu/faible dépense, etc.).









------------------








## Étape 7 — Expérimentations : méthode du coude et silhouette
**Exemple de code**


```python
inertia = []
K_range = range(2, 11)
for k2 in K_range:
    km2 = KMeans(n_clusters=k2, random_state=42)
    km2.fit(X_scaled)
    inertia.append(km2.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.title("Méthode du coude")
plt.xlabel("Nombre de clusters k")
plt.ylabel("Inertia")
plt.show()
```

**AFFICHAGE**



<img width="1025" height="425" alt="image" src="https://github.com/user-attachments/assets/7a6ef2d9-0a81-4cce-b781-ceb0ff3ce4f2" />



**Interprétation**

Le “coude” du graphique indique le k optimal (ex. 3 ou 5 selon les données).









--------------------








## Étape 8 — Réponses aux questions (Étape 2 du TP)


**Question	Réponse**


Quelles sont les différences entre les clusters trouvés et les classes réelles ?	K-Means ne connaît pas les classes réelles ; il regroupe les données selon leurs similarités numériques. Les clusters peuvent donc ne pas correspondre exactement aux classes humaines.
Quel est l’impact de la normalisation ?	Elle met toutes les variables à la même échelle, évitant qu’une variable (comme le revenu) domine les autres dans la distance euclidienne.
Comment choisir le nombre optimal de clusters (k) ?	On utilise la méthode du coude (elbow) ou le silhouette score : le “coude” indique où ajouter un cluster n’apporte plus beaucoup d’amélioration.




| Étape | Action                                   | Bibliothèque         |
| :---- | :--------------------------------------- | :------------------- |
| **1** | Chargement des données                   | pandas               |
| **2** | Prétraitement (nettoyage, normalisation) | pandas, scikit-learn |
| **3** | Visualisation initiale                   | matplotlib, seaborn  |
| **4** | Clustering K-Means                       | scikit-learn         |
| **5** | Évaluation des clusters                  | scikit-learn         |
| **6** | Visualisation finale                     | matplotlib, seaborn  |




## Conclusion

Ce TP a permis de mettre en pratique les principales étapes de l’apprentissage non supervisé à travers l’algorithme K-Means.
Nous avons appris à :

Explorer et préparer un dataset (vérification des valeurs manquantes, sélection et normalisation des variables pertinentes).

Visualiser les données pour repérer intuitivement des regroupements.

Appliquer l’algorithme K-Means pour segmenter les données selon leurs similarités.

Évaluer la qualité des clusters grâce au Silhouette Score et à la méthode du coude, permettant de déterminer le nombre optimal de clusters.

Interpréter les résultats et comprendre l’influence de la normalisation sur la qualité du clustering.

Ce travail illustre la puissance du clustering dans la découverte de structures cachées au sein des données.
K-Means, bien que simple, constitue une base essentielle pour aborder des techniques plus avancées de segmentation non supervisée, comme DBSCAN ou Gaussian Mixture Models (GMM).
