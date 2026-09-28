# FHO3 ThreadMap fragments — audit post-shake, 28 septembre 2026

## But et invariants

Ce prototype expose les tranches par stage d'une contrainte `ThreadMap`
agrégée après le `shake` Bioptim. Il ne modifie ni le graphe NLP MX, ni le
vecteur canonique `g`, ni les callbacks IPOPT : l'agrégat parent reste la seule
contrainte canonique. Chaque fragment conserve :

- le terme parent ;
- l'offset de lignes dans ce parent ;
- les lignes canoniques `g` résolues après `shake` ;
- les fonctions locales exactes valeur/Jacobienne/Hessienne du Lagrangien.

La tranche permet donc en principe : un noyau C par signature, un `map` sur
les stages, puis un scatter-add exact dans une future callback Hessienne. Le
FHO global reste MX et non compilé.

## Validation unitaire

Le test `test_thread_map_fragments_resolve_to_slices_of_their_aggregate_parent_rows`
vérifie deux fragments quadratiques : parent `g=[x0²,x1²]`, fragments affectés
aux lignes canoniques `[0,1)` et `[1,2)`, valeurs exactes 9 et 25. Avec les
tests de registre et d'audit NLP : **9 passed**.

## Mesure FHO3 réelle

Référence FHO3 à 0,30 Nm :

| Mesure | Valeur |
|---|---:|
| Variables / contraintes | 12 263 / 11 911 |
| Hessienne native, nnz triangulaires | 49 077 |
| Groupe répété observable avant fragments | 29 stages × 12 nnz locaux |
| Couverture brute de ce groupe | 348 / 49 077 = **0,709 %** |
| Hessienne MX map, 29 stages | 0,422 ms |
| Noyau C map, 29 stages | 0,379 ms |
| Accélération locale C | **×1,11** |

Les données de référence sont dans
`local-results/fho3-postshake-registry-benchmark-20260927-v10/report.json`.

## Résultat négatif important

Le premier audit complet avec tous les fragments ThreadMap ne termine pas dans
la fenêtre de 30 s (contre 16,8 s pour le registre sans fragments). Réutiliser
l'expression parent déjà `shake` évite de secouer le graphe 29 fois, mais la
localisation MX par tranche reste dominante.

Ce n'est pas un échec de solveur, mais une indication de conception :
**l'extraction des fragments ne doit pas être faite a posteriori sur le grand
graphe MX**. Bioptim doit conserver les fragments `ThreadMap` et leurs
indices globaux au moment de la construction des pénalités, avant agrégation
et shake. Le registre post-shake est la preuve du contrat de provenance, pas
encore une méthode de préparation rapide.

Enfin, 0,709 % de couverture signifie qu'un gain local de ×1,11 ne peut pas
réduire sensiblement le temps total de Hessienne FHO3. Il serait prématuré de
brancher cette callback à IPOPT. Le prochain essai utile doit viser les
fragments dynamiques/collocation qui couvrent une part substantielle des
49 077 entrées, avec la provenance pré-agrégation.
