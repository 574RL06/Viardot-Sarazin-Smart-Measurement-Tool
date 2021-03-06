# Viardot-Sarazin-Smart-Measurement-Tool
Outil de mesure intelligent par Viardot-Sarazin
à la demande de prof. Antoine Ferreira, maître de conférence.

(Cliquez sur Code puis Download ZIP pour récupérer l'ensemble du projet sur votre ordinateur)

## Membres
- Hugo Sarazin, 4ème année MRI - EA Systèmes Avancés
- Jean-François Viardot, 4ème année MRI - EA Systèmes Avancés

## Introduction
Les outils mathématiques qu'offrent Newton ou encore Lagrange au regard des algorithmes d'interpolation semblent aujourd'hui presque dépréciés. Face à la puissance des algorithmes d'intelligence artificielle de machine learning et de deep learning, les mathématiques standards ont des airs de lointains souvenirs.

## Résumé
Le protocole de mesure expérimentale a été amélioré et automatisé. Avec une vitesse d'échantillonage d'une mesure par seconde, le robot dressera une dataset complète en 12 heures et 9 minutes. Il apparait d'autant plus évident que la mesure manuelle est totalement dépréciée.

La dataset a été structurée afin d'être injectée dans le module Python TensorFlow.

La finalité est a posterio de pouvoir donner à TensorFlow un ensemble de paramètres tels que la position de l'espace, le champ magnétique et l'orientation du champ magnétique souhaité. L'algorithme donnera à l'utilisateur (voire en commande directe sur le système) les valeurs de position des bobines afin d'obtenir la sortie souhaitée.

## Conclusion
Nous avons travaillé avec une approche quasi-industrielle dans le cadre de cette étude. Nous n’avons pas cherché à faire évoluer la technique sur le plan mathématique mais avons optimisé l’expérience afin d’obtenir un rendement de sortie.

## Remerciement
Nous remercions :
 - M. Ruipeng Chen, doctorant, pour ses explications.
 - M. Antoine Ferreira, maîre de conférence, pour ses enseignements et son accompagnement.
 - M. Benoit Magnain, maître de conférence, pour ses enseignements de Robotique.
 - M. Adel Hafiane, maître de conférence, pour ses enseignements de Classification et Analyse des données.
