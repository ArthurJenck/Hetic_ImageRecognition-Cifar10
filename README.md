# Hetic_ImageRecognition-Cifar10

Modèle de réseau de neurones convolutif (CNN) pour la classification d'images sur le dataset CIFAR-10.

## Dataset

CIFAR-10 contient 60 000 images en couleur de 32x32 pixels réparties en 10 classes :

- avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, navire, camion

## Architecture du modèle

- Input : 32x32x3
- Conv2D : 32 filtres 3x3, ReLU
- MaxPooling2D : 2x2
- Conv2D : 64 filtres 3x3, ReLU
- MaxPooling2D : 2x2
- Conv2D : 64 filtres 3x3, ReLU
- Flatten
- Dense : 64 neurones, ReLU
- Dense : 10 neurones, Softmax

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution

```bash
python cifar10_cnn.py
```

Le script effectue :

1. Chargement et visualisation des données
2. Construction du modèle
3. Entraînement sur 10 epochs
4. Évaluation des performances
5. Génération des courbes d'apprentissage
6. Affichage de la matrice de confusion
7. Sauvegarde du modèle entraîné

## Modèle sauvegardé

Le modèle entraîné est sauvegardé dans `cifar10_cnn_model.keras`.
