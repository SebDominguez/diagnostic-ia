# Conformité RGPD & responsabilité juridique

## 1. Nature des données

Les données traitées (constantes vitales, antécédents, récit de symptômes) sont des
**données de santé** : une **catégorie particulière** au sens de l'**article 9 du RGPD**,
dont le traitement est interdit par principe sauf exception (consentement explicite,
intérêt vital, ou mission de soins par un professionnel de santé soumis au secret).

## 2. Principes appliqués dans le projet

| Principe RGPD | Mise en œuvre |
|---|---|
| **Minimisation** (Art. 5) | Le modèle de production (Scénario 2) **n'utilise pas** `sexe` ni `zone_vie` : variables sensibles/discriminantes retirées sans perte de performance. L'API ne les demande donc pas. |
| **Pseudonymisation** (Art. 4) | `patient_id` est conservé pour la traçabilité mais **exclu de la modélisation**. À renforcer en prod : table de correspondance chiffrée et séparée. |
| **Exactitude** (Art. 5) | Règles métier + détection d'outliers + imputation lors du nettoyage. |
| **Traçabilité / journalisation** (Art. 30) | Chaque inférence est journalisée (entrée, sortie, date, session) dans `predictions.jsonl` — registre des traitements et auditabilité. |
| **Limitation de conservation** (Art. 5) | À définir : purge automatique des logs au-delà de la durée nécessaire (ex. 1 an). |
| **Sécurité** (Art. 32) | À prévoir en prod : chiffrement at-rest et in-transit (HTTPS), contrôle d'accès, logs d'accès. |
| **Droits des personnes** (Art. 15-17) | Droit d'accès et d'effacement : le `patient_id` permet de retrouver et supprimer les enregistrements d'une personne. |

## 3. Responsabilité juridique en cas d'erreur de classification

- **Outil d'aide à la décision, pas de décision autonome.** Le système **assiste** le tri ;
  la décision finale reste celle du **professionnel de santé**, qui en porte la responsabilité.
  Un **humain dans la boucle** est obligatoire (cf. Art. 22 RGPD : droit de ne pas faire
  l'objet d'une décision **entièrement automatisée** à effet significatif).

- **Erreur critique = sous-triage.** Le risque juridique majeur est le **faux négatif sur
  l'urgence vitale** (classe 2 prédite 0/1). Le projet l'adresse explicitement via la
  **décision par coût minimal** qui pénalise lourdement ce cas (cf. notebook §20).

- **Chaîne de responsabilité.** Le **fournisseur** du logiciel répond de la conformité, de
  la documentation et des performances annoncées ; l'**établissement** répond de l'usage et
  de la supervision humaine. La **journalisation** des inférences est l'élément de preuve en
  cas de litige.

- **Cadre applicable.** Au-delà du RGPD : **AI Act** (un système de triage médical est à
  **haut risque** → exigences de transparence, supervision humaine, gestion des risques) et
  réglementation **dispositif médical** (marquage CE) selon l'usage revendiqué.

> Synthèse : minimisation des données sensibles, traçabilité complète, supervision humaine
> obligatoire, et optimisation explicite contre l'erreur la plus grave (sous-triage vital).
