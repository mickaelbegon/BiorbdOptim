# Prototype de fragments `ThreadMap` avant agrégation — 28 septembre 2026

## Décision de conception

Le point d'insertion est `generic_get_all_penalties`, immédiatement avant
l'appel agrégé à `penalty.weighted_function[0]`. Chaque pénalité multi-thread
conserve déjà `weighted_function_non_threaded[0]`: c'est exactement le noyau
scalaire du stage, avant `Function.map` puis concaténation dans le grand graphe
MX.

Le prototype opt-in appelle ce noyau une fois par `node_idx` et enregistre :

- le graphe scalaire original du stage ;
- le parent agrégé canonique ;
- son décalage de lignes dans le parent ;
- après dispatch, l'intervalle canonique global `g` ;
- les colonnes actives de `x` et les sparsités Jacobienne/Hessienne exactes.

Le graphe et les callbacks soumis à IPOPT restent strictement inchangés. Les
fragments ne sont construits que par
`SolverInterface.build_post_shake_penalty_registry()`.

## Éviter le coût du post-shake par stage

Le prototype précédent découpait le parent déjà `shake`, puis localisait les
tranches : cette reconstruction MX était le coût dominant. Le nouveau chemin
ne découpe jamais ce parent. Pour un fragment, il applique directement la
substitution des durées de phase fixes, qui est la seule transformation de
`_shake_penalties_tree`; le `Function` temporaire CasADi n'est donc pas créé
des centaines de fois. Le parent continue d'employer le `shake` normal, ce qui
conserve le contrat canonique.

## Validation exacte

Sur le pendule en collocation directe MX (3 intervalles, degré 3, 2 threads),
les trois fragments `STATE_CONTINUITY` sont capturés avant `map` :

| fragment | lignes `g` canonique |
|---|---:|
| 0 | `[0, 16)` |
| 1 | `[16, 32)` |
| 2 | `[32, 48)` |

Le test d'intégration vérifie, au point initial, que chaque valeur, Jacobienne
et Hessienne locale (multiplicateurs unitaires) est égale au bloc global
correspondant. Les tests de registre couvrent aussi la provenance parent et le
cas de graphes structurellement indépendants mais algébriquement équivalents.

Commande :

```bash
PYTHONPATH=. python -m pytest \
  tests/shard6/test_post_shake_penalty_registry.py \
  tests/shard1/test_initial_nlp_audit.py::test_post_shake_penalty_registry_uses_final_canonical_constraint_rows \
  tests/shard1/test_initial_nlp_audit.py::test_pre_shake_thread_map_fragments_use_scalar_stage_graphs_and_exact_canonical_rows -q
```

Résultat : `6 passed`.

## Portée FHO

L'audit FHO3 séparé attribue environ 98,845 % des couples non nuls de la
Hessienne native à `STATE_CONTINUITY`. Ces fragments sont donc la bonne cible,
contrairement au premier groupe post-shake de faible couverture. Ce prototype
ne branche encore aucune callback IPOPT : la prochaine mesure doit comparer la
préparation du registre FHO3 et un assemblage/scatter exact par paquets à la
Hessienne monolithique native.
