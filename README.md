# Multi-View Molecular Learning for NMR Prediction

## Introduction

MultiNMR is a software package for processing and analyzing NMR spectra.

## Fingerprints

- `Morgan`: Morgan fingerprints are circular fingerprints that represent the presence or absence of a substructure in a molecule. They are based on the concept of Morgan fingerprints, which are based on the distance between atoms in a molecule.
  
- `ErG`: ErG fingerprints are circular fingerprints that represent the presence or absence of a substructure in a molecule. They are based on the concept of Ertl-Gordon fingerprints, which are based on the distance between atoms in a molecule and the number of bonds between them.

- `PubChem`: PubChem fingerprints are circular fingerprints that represent the presence or absence of a substructure in a molecule. They are based on the concept of PubChem fingerprints, which are based on the distance between atoms in a molecule and the number of bonds between them.


## Node Features

- `atomic number`: atomic number of the atom

- `is aromatic`: whether the atom is aromatic or not

- `chirality`: whether the atom is chiral or not

- `hybridization`: whether the atom is single, double or triple bonded

- `degree`: number of bonds to other atoms

- `formal charge`: the net charge of the atom

- `no. hydrogens`: number of hydrogens on the atom

- `no. radical electrons`: number of radical electrons on the atom

- `implicit valence`: the number of implicit valence electrons on the atom

- `no. rings for each ring size`: number of rings of each size attached to the atom

## Edge Features

- `bond type`: the type of bond (single, double, triple, aromatic)

- `stereochemistry`: whether the bond is cis or trans

- `is conjugated`: whether the bond is conjugated or not

- `is in ring`: whether the bond is in a ring or not
