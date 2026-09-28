# Protocole isolé : `H_other` + continuité C sparse, de FHO3 à FHO100

## Objectif

Évaluer une future voie hybride sans jamais risquer une campagne :

```
H_L = H_other (CasADi natif) + H_STATE_CONTINUITY (C exact, sparse)
```

Le benchmark construit des callbacks CasADi, puis intercepte l'appel IPOPT
avant toute itération. Il ne produit donc ni solution, ni état terminal, ni
écriture dans les résultats de campagne.

## Étape obligatoire FHO3

Le candidat doit être comparé au NLP MX natif, au même point et mêmes
multiplicateurs. Mesures requises :

| Signal | Critère de passage |
|---|---:|
| `nlp_f`, `nlp_g`, `nlp_jac_g`, `nlp_hess_l` | erreur max ≤ `1e-10` |
| quatre sparsités de sortie | mêmes `nnz` |
| `nlp_hess_l` à chaud | accélération ≥ ×1,05 |
| construction froide / chaude | rapportée séparément, jamais mélangée aux callbacks |

Avant toute mesure de vitesse, le candidat hybride doit aussi respecter
l'ABI native `nlp_hess_l(x, p, sigma, lambda_g)`. Les quatre noms, formes et
patrons sparse d'entrée, ainsi que le patron de sortie triangulaire, sont
comparés strictement. En particulier, un paramètre absent peut être de
sparsité `0×0` : le remplacer par un vecteur `0×1` est un échec ABI même si les
deux contiennent zéro valeur. Les lignes de continuité sont perturbées dans le
vecteur global `lambda_g` selon leurs indices canoniques, puis la réponse de
la Hessienne est comparée. Cela interdit d'utiliser par erreur un ordre de
multiplicateurs local à un paquet.

`scripts/hybrid_hessian_contract.py` fournit cet audit indépendant de la
construction de `H_other`; il doit être appelé par le futur harness hybride.

L'accélération minimale est volontairement basse : FHO3 sert à établir une
preuve de non-régression. Une voie qui ne dépasse pas ce seuil ne justifie pas
l'allocation mémoire et le temps de construction de FHO100.

## Étape FHO100, seulement après passage FHO3

Le harness lance FHO100 exclusivement lorsque la porte FHO3 est validée. Les
métriques supplémentaires à reporter sont : dimension `x/g`, nnz Jacobienne et
Hessienne, nombre/taille des paquets, temps de construction native/C froide et
chaude, temps médian de `nlp_jac_g` et `nlp_hess_l`, RSS maximal, et erreurs
exactes. Les mêmes seuils d'égalité s'appliquent. Aucun appel à IPOPT n'est
autorisé dans cette phase.

## Commande

```bash
python scripts/benchmark_hybrid_hessian_protocol.py \
  --fho3-command-log /chemin/FHO3-command.json \
  --fho100-command-log /chemin/FHO100-command.json \
  --output /tmp/hybrid-hessian-protocol --packet-rows 32
```

Le rapport `protocol-report.json` indique explicitement `fho100_started`.
Quand FHO3 échoue une égalité ou le seuil de gain, sa valeur reste `false` et
la commande FHO100 n'est jamais exécutée.
