# POC — `External` C exact sous un `ThreadMap`

## But

Tester l'intégration la moins intrusive possible : le FHO global reste un
graphe MX CasADi normal, non compilé. Pour une famille choisie, seul le noyau
scalaire sous `Function.map` est remplacé par un `External` C. CasADi conserve
donc la construction des callbacks IPOPT globaux (`nlp_f`, `nlp_g`,
`nlp_jac_g`, `nlp_hess_l`).

Le POC est désactivé par défaut. Il s'active explicitement avant la
construction de l'NLP :

```python
interface.enable_compiled_thread_map_external("STATE_CONTINUITY", cache_dir)
```

## Dérivées exactes

La bibliothèque C contient : primal, Jacobienne, `fwd1`, `adj1`,
`fwd1_adj1` et `adj1_fwd1`. Le primal seul est insuffisant : CasADi doit
pouvoir différencier l'appel `External` en avant et en arrière pour former la
Hessienne lagrangienne globale. Le test unitaire compare valeur, Jacobienne et
Hessienne de l'External à la fonction MX native avec une erreur nulle.

## Mesure FHO3

`scripts/benchmark_external_state_continuity_global.py` reconstruit le FHO3
historique et intercepte l'appel IPOPT avant toute itération. Il compare les
quatre callbacks globaux sur le même point et compare les Hessiennes en format
sparse afin de ne pas densifier la matrice 12 263 x 12 263.

Le coût de compilation est lui-même un résultat : compiler le noyau scalaire
de continuité complet et ses dérivées produit environ 2,7 Mo de C et requiert
un cache persistant. Ce coût est unique par structure de NLP mais interdit de
reconstruire ce noyau à chaque horizon RHO/FHO. La mesure finale doit donc
séparer construction/compilation, puis évaluation à chaud.

## Limites

Ce POC ne modifie aucune campagne, n'installe aucun callback IPOPT manuel et
ne remplace pas encore le registre pré-`shake`. Il vérifie seulement que
l'option la plus directe — `External` exact sous le `map` — préserve les
dérivées globales. Une intégration production exigera un cache signé/verrouillé
et des mesures à chaud qui dépassent le coût d'appel ABI.

## Verdict FHO3 du 28 septembre

Le test complet a rejeté cette voie directe. Le graphe se construit et les
sorties objective/contraintes restent compatibles, mais le `External` avec
Jacobienne personnalisée perd le patron de Hessienne de CasADi :

| callback | natif | External | résultat |
|---|---:|---:|---|
| `nlp_hess_l` nnz | 49 077 | 1 156 389 | patron densifié |
| `nlp_hess_l` | 0,164 s | 1,466 s | ×0,112 |
| construction NLP | 5,86 s | 211,69 s | non amortissable ici |

La compilation/assemblage a culminé à 3,9 Gio. Les évaluations `g` et `J`
ont été faites au zéro symbolique et contenaient des `NaN` issus du modèle ;
elles ne constituent donc pas un certificat numérique. Le gate FHO3 rejette
désormais explicitement toute erreur non finie. La conclusion est **NO-GO pour
l'External pré-map monolithique** ; la cible reste `H_other` natif plus une
continuité C assemblée directement dans le patron sparse original.
