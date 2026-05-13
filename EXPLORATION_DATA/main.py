import math
import pandas
import matplotlib.pyplot as plt


# 1 - Charger dataset
data = pandas.read_csv("mots.csv")

mots = data["mot"].tolist()
categories = data["categorie"].tolist()

# 2 - Liste unique des catégories
liste_categories = []

for categorie in categories:
    if categorie not in liste_categories:
        liste_categories.append(categorie)

print("Catégories uniques :")
print(liste_categories)

# 3 - Transformer catégories en vecteurs (One Hot Encoding)
vecteurs = []
for categorie in categories:
    vecteur = []
    for cat in liste_categories:
        if categorie == cat:
            vecteur.append(1)
        else:
            vecteur.append(0)
    vecteurs.append(vecteur)

# 4 - Affichage des vecteurs

print("\nVecteurs :")

for i in range(len(mots)):
    print(mots[i], "=", vecteurs[i])

# 5 - Calcul des distances Euclidiennes

print("\nDistances Euclidiennes :")

for i in range(len(vecteurs)):
    for j in range(i + 1, len(vecteurs)):
        somme = 0
        for k in range(len(liste_categories)):
            somme += (vecteurs[i][k] - vecteurs[j][k]) ** 2
        distance = math.sqrt(somme)
        print(mots[i], "-", mots[j], "=", round(distance, 2))

#5- visualisation -afficher le graphe
plt.figure(figsize=(10,7))
for i in range(len(mots)):
    x=vecteurs[i][0]
    y=vecteurs[i][1]
    plt.scatter(x,y)
    plt.text(x+0.1, y+0.1, mots[i])
plt.title("Representations schematique des mots")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid()
plt.show()

