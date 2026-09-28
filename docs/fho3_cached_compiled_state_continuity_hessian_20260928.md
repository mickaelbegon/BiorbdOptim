# FHO3 — Hessienne exacte `STATE_CONTINUITY` compilée et assemblée sparse

## Périmètre

Ce test reste entièrement hors solveur : il reconstruit le FHO3 historique
(0,30 Nm, collocation MX), capture `nlp_hess_l`, mais n'installe aucun callback
et ne lance aucune itération IPOPT. Les multiplicateurs sont non nuls seulement
sur les lignes canoniques `STATE_CONTINUITY`, avec `sigma=0`.

Le noyau C est généré à partir de **la fonction de Hessienne lagrangienne
locale exacte existante**, donc inclut les dérivées secondes ; ce n'est pas une
compilation du seul primal. La conversion VM/C est vérifiée avant le `map` :
erreur maximale **0**. Chaque paquet reconstruit ensuite les 49 077 valeurs
stockées de `nlp_hess_l` avec erreur maximale **0** et zéro entrée au-dessus de
`1e-10`.

## Cache structurel

La structure est construite une fois par NLP, puis réutilisable pour toutes les
évaluations : indices de rassemblement de `x` et `lambda`, offsets des nnz
locaux et slots triangulaires globaux. L'assemblage ne densifie plus la sortie
`158 x (158*90)` : il lit directement `mapped_value.nonzeros()` et applique un
`numpy.bincount` sur les slots pré-calculés.

| Élément FHO3 | Valeur |
|---|---:|
| Variables / contraintes | 12 263 / 11 911 |
| nnz Hessienne triangulaire globale | 49 077 |
| Fragments / variables locales / lignes par fragment | 90 / 158 / 132 |
| nnz Hessienne locale | 1 004 |
| Registre brut, une fois | 0,349 s |
| Paquet compact, une fois | 18,986 s |
| Plan sparse, une fois | 0,118 s |
| Génération + compilation C `-O0`, une fois | 10,734 s |

Le coût du paquet compact doit donc être amorti sur les évaluations IPOPT ; il
ne doit jamais être reconstruit dans une callback.

## Temps par évaluation (médian, secondes)

| Kernel | Workers | Évaluation locale | Gather + scatter sparse | Total callback simulée | `nlp_hess_l` natif | Gain local |
|---|---:|---:|---:|---:|---:|---:|
| VM | 1 | 1,057 | 0,007 | 1,064 | 0,206 | 0,19× |
| VM | 2 | 0,517 | 0,008 | 0,525 | 0,206 | 0,39× |
| VM | 3 | 0,361 | 0,007 | 0,368 | 0,206 | 0,56× |
| VM | 12 | 0,142 | 0,008 | 0,150 | 0,206 | 1,38× |
| C exact | 1 | 0,187 | 0,008 | 0,194 | 0,206 | 1,06× |
| C exact | 2 | 0,091 | 0,008 | 0,099 | 0,206 | 2,09× |
| C exact | 3 | 0,061 | 0,008 | 0,068 | 0,206 | 3,03× |
| C exact | 12 | 0,036 | 0,008 | **0,045** | 0,206 | **4,63×** |

La ligne C/12 est une accélération de la contribution `STATE_CONTINUITY`
isolée, pas encore du FHO complet : objectif et autres contraintes restent à
reconstruire ou conserver dans le chemin natif avant tout callback réel.

## Décision et suite

Le résultat rend la piste prometteuse : la compilation exacte et le cache sparse
suppriment les deux goulots du prototype précédent (VM local et dense scatter).
La suite sûre est un callback expérimental **hybride** qui injecte seulement
cette famille, compare à chaque appel le vecteur de Hessienne complet au chemin
natif, puis mesure un solve complet. Il faut également évaluer `-O1`/`-O2` :
`-O0` compile de façon fiable en ~11 s. À titre indicatif, `-O1` a produit une
bibliothèque de 2,1 Mo mais a demandé 54,4 s et 1,7 Gio (le source exact fait
~13 Mo) ; il faut donc justifier toute optimisation de compilation par une
mesure d'exécution supplémentaire.

## Reproduction

```bash
PYTHONPATH=.:/home/mickaelbegon/Documents/Kevin/cocofest-pedalage \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 12-23 python scripts/benchmark_compiled_state_continuity_packet_hessian.py \
  --command-log /home/mickaelbegon/Documents/Kevin/cocofest-pedalage/local-results/fho3-postshake-packet-audit-20260927-v3/command.json \
  --output /tmp/fho3-compiled-state-continuity \
  --hsl-library /home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so \
  --repeats 3 --workers 1 2 3 12 --compile-local-kernel --compiler-optimization=-O0
```

Mesures brutes :
`local-results/fho3-compiled-state-continuity-packet-hessian-20260928-v9/report.json`.
