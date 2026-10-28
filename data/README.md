# Source CSV datasets

This directory contains the CSV datasets selected for this release:

- `CatPred-DB/`: `kcat-random_train.csv`, `kcat-random_val.csv`, and `kcat-random_test.csv`.
- `PMAK-DB/catpred/`: `train.csv`, `test.csv`, and `OOD.csv`.
- `PMAK-DB/turnup/cold_enzyme/`: five official train/held-out fold pairs.
- `PMAK-DB/turnup/cold_reaction/`: five official train/held-out fold pairs.

There are 26 source CSV files. The supplied files and their splits are
preserved during code reorganization. Preprocessing writes generated arrays
under `../features/`, and never replaces these CSV files.

The common input columns are `sequence`, `log10kcat_max` and reaction SMILES.
When `reaction_smiles` is empty, preprocessing reconstructs it from
`reactant_smiles` and `product_smiles`; this is required by the supplied
TurnUp cold-reaction files. The target is log10 catalytic rate in s^-1.

CatPred uses its supplied validation split for checkpoint selection. PMAK's
`test.csv` and `OOD.csv`, and TurnUp's `kcat_val_fold_*` files, are held out
until final-epoch training has completed in their respective trainers.

The upstream CatPred source is maranasgroup/CatPred-DB. Preserve the original
data attribution and applicable terms when redistributing these datasets.
