# FHO3 — paquet Hessienne `STATE_CONTINUITY` exact, 28 septembre 2026

## But et périmètre

Ce test construit le FHO3 historique (0,30 Nm, MX non compilé), mais ne lance
aucune itération IPOPT et ne remplace aucun callback. Il capture les fragments
`STATE_CONTINUITY` avant `Function.map`, applique un noyau local exact au moyen
de `Function.map(..., "thread", n)`, puis fait un *scatter-add* dans le
pattern triangulaire de `nlp_hess_l`.

La référence est `nlp_hess_l(x0, p, sigma=0, lambda)`, avec multiplicateurs
non nuls uniquement sur les lignes canoniques `STATE_CONTINUITY`. Le vecteur
initial physique `x0` est utilisé : le vecteur nul est hors domaine FES et
produit des NaN légitimes dans des dérivées secondes.

## Dimensions et exactitude

| Élément | Valeur |
|---|---:|
| Variables / contraintes FHO3 | 12 263 / 11 911 |
| nnz triangulaires Hessienne globale | 49 077 |
| Fragments `STATE_CONTINUITY` | 90 |
| Variables locales / lignes `g` par fragment | 158 / 132 |
| nnz Hessienne locale | 1 004 |
| Construction registre brut | 0,686 s |
| Construction paquet compact | 22,941 s |
| Erreur maximale packet − `nlp_hess_l` | **0.0** |

Le *scatter* est donc exactement équivalent au bloc Hessien monolithique pour
ce choix de multiplicateurs, y compris les indices globaux et les lignes `g`.

## Temps mesurés

Les bibliothèques numériques sont mono-thread ; seul le `map` CasADi utilise
le nombre de workers indiqué.

| Workers | Évaluation locale | Scatter-add | Total packet | `nlp_hess_l` natif | Rapport natif / packet |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,270 s | 0,038 s | 1,307 s | 0,237 s | 0,18× |
| 2 | 0,605 s | 0,041 s | 0,646 s | 0,237 s | 0,37× |
| 3 | 0,464 s | 0,038 s | 0,502 s | 0,237 s | 0,47× |
| 12 | 0,180 s | 0,047 s | 0,227 s | 0,237 s | **1,04×** |

Les valeurs viennent de
`local-results/fho3-state-continuity-packet-hessian-20260928-v9/report.json`.
Il s'agit d'une répétition unique : la conclusion est qualitative, pas un
benchmark statistique définitif.

## Décision

Le contrat de produit est réaliste : les fragments pré-map et le scatter
produisent une Hessienne exacte sans toucher au FHO MX. En revanche, ce
premier noyau VM n'apporte qu'un gain marginal à 12 workers (environ 4 % sur
la seule contribution `STATE_CONTINUITY`), avant même d'ajouter objectif et
autres contraintes. Il n'est donc **pas justifié d'installer un callback
IPOPT** à ce stade.

La prochaine expérimentation utile est de compiler le noyau local exact et
ses dérivées, puis de mesurer le même paquet avec un assemblage sparse sans
boucle Python. La construction compacte à 23 s doit aussi être mise en cache
par structure NLP, jamais reconstruite à chaque évaluation.

## Reproductibilité

```bash
PYTHONPATH=.:/home/mickaelbegon/Documents/Kevin/cocofest-pedalage \
  python scripts/benchmark_state_continuity_packet_hessian.py \
  --command-log /home/mickaelbegon/Documents/Kevin/cocofest-pedalage/local-results/fho3-postshake-packet-audit-20260927-v3/command.json \
  --output /tmp/fho3-state-continuity-packet \
  --hsl-library /home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so \
  --repeats 3 --workers 1 2 3 12
```

Le script est uniquement un audit : il ne lance pas IPOPT, ne modifie pas le
callback et ne produit aucune campagne.
