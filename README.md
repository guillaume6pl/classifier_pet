classifieur_pet.py
======

## Log
Using TensorFlow backend.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
./extract-features.py:27: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=Tensor("fc..., inputs=Tensor("in...)`
  model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

## Objectif
- Réaliser l’apprentissage d’un classifieur d’images: indique de manière automatique quel est le contenu principal d’une image, à partir d’une liste de classes possibles.
- 2 classes C = ["chien", "chat"]
- fonction f tel que f(<image>,C,Θ) = "<chat|chien>"


## Démarche
1. Télécharger un dataset constitué de photos de chats ou chiens
2. Prototyper en local une application Spark avec un script python classifieur_pet.py qui récupère une partie du dataset et crée des modèles de ML. Cela inclut :
    * séparer le dataset en jeux d'apprentissage et de test
    * extraire les features des images
    * créer les modèles (classifieurs d'image)
        - "1 vs. 1" pour toutes les paires de classes (X, Y) possibles : on part du principe que l'on sait déjà que les images appartiennent à  X ou Y)
        - "1 vs All" : permet, à partir de n'importe quelle image de test, de prédire si elle appartient à la classe X ou non
    * mesurer les performances du modèle sur les données de test
3. Déployer l'application sur un cluster de calcul AWS pour réaliser l'apprentissage complet
    * récolter les performances de classification dans S3
    * détecter les goulots d'étranglement de votre application et proposer si possible des solutions permettant de s'en affranchir
4. Aller plus loin :
    * Amélioration du modèle
        - classifieur type
        - features

## Considerations about this ML case
**Dataset**
[Oxford IIIT-Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
- 7390 images de chats ou chiens, 37 classes différentes correspondant à une race différente de chien et de chat
- dataset de 755 Mo

**Learning and test data**
- le dataset est scindé en 2 groupes :
    1. 100 premières de chaque classe > learning data (soit 3700 images en tout)
    2. autres images > test data (soit 3690 images en tout)

**Apprentissage du modèle de classification**
- Charactéristiques des features: sous la forme d’un tableau de 4096 valeurs flottantes au format JSON
- Méthode utilisée: réseaux de neurones convolutionnels (CNN) sous la forme d'une architecture deep learning. Pour chaque image :
    1. pour chaque image appliquer un CNN - fonction SVMWithSGD (Support Vector Machine with Stochastic Gradient Descent)
    2. extraire les valeurs de sortie d’une des couches intermédiaires
    3. extraire les features sont issues de l'avant-dernière couche du réseau de neurones


## Architecture de classifieur_pet.py



## Instructions en local
0. Récupérer les fichiers de la collection

1. Démarrer hadoop
```
$ start-dfs.sh
```

2. Extract features. Choose appropriate command regarding os limitations/config
```
$ cd <path_to_scripts>
option 1:
$ ./extract-features.py ./input/images/*.jpg
option 2:
$ printf '%s\0' ./input/images/*.jpg | xargs -0 ./extract-features.py
option 3:
$ find ./input/images/ -maxdepth 1 -type f -name '*.jpg' -exec python ./extract-features.py '{}' +
```

3. Execute main script
```
$ cd <path_to_scripts>
$ spark-submit classifieur_pet.py "<path/to/features/>"
```

4. Results: "best_classifiers.json" contains evaluation results for each best model



## Résultats - Précision des modèle de classification d'image 1vs1 & 1vsAll
- 1vsAll                                            95%
- Wheaten Terrier vs Yorkshire Terrier              98% possible
- ...                                               96%


## Livrable
- script Python contenant le code de l'application
- Une image représentant l'évolution des performances du modèle en fonction du nombre d'images d'apprentissage


## Soutenance
- Présentation en tant que data architect freelance ayant mis en place une application d'identification de photos d'animaux pour votre client, la fondation 30 Millions d'Amis
- Public sera constitué des data scientists et des data architects de la fondation (le mentor)
- Soutenance durera 25 minutes + 5 minutes de questions/réponses
    * Résumé du contexte, présentation des enjeux 	           10 min
    * Présentation des choix techniques réalisés 	           10 min
    * Présentation des performances obtenues 	               5 min
    * Questions 	                                           5 min
- Compétences à valider:
    * notion de RDDs
    * Débugger/Accélérer un programme distribué
    * Déployer et administrer une plateforme de calcul distribué dans le cloud
    * Réaliser l'apprentissage distribué d'un modèle de machine learning
    * Réaliser un calcul distribué avec Spark


## Version
* Version 1.0

## Contact
* e-mail: guillaume.attard@gmail.com
* Twitter: [@guillaumeattard](https://twitter.com/guillaumeattard)
