# BioChem-Hackathon
This is our submission for BioChem AI hackathon conducted by ChemBio AI under Prometeo'24

## Problem Statement
Build a seq to seq based model which could predict the reaction conditions for the reaction i.e. given reaction smiles as an input it should be able to predict the conditions.

## Problem Description 
To give an overview of the problem statement, We are given a dataset that contains the organic reaction in smiley format. The conditions on which the predicton is required are Reagent, Solvent and Catalyst based on the Given Origanic reaction

## Data Description 
1. canonic_rxn: This column contains the canonical representation of chemical reactions using SMILES (Simplified Molecular Input Line Entry System) strings. Each row represents a distinct chemical transformation, capturing the reactants and products involved. 

2. rxnmapper_aam: This column encodes atom-to-atom mappings (AAM) for the reaction. AAM is a technique used to establish correspondence between atoms in reactants and products, facilitating the tracking of atom transformations during a reaction. 

3. Reagent: This column details the reagents utilized in the chemical reactions. Reagents are chemical substances introduced into a reaction to initiate or facilitate the transformation. 

4. Solvent: For each reaction, the solvents used are listed in the solvent column. Reaction conditions are greatly influenced by solvents, which also have an impact on reaction rates and results. 

5. Catalyst: Substances known as catalysts quicken reactions without changing permanently on their own. Understanding catalysts helps one gain understanding of the efficiency and mechanism of reactions. 

6. Yield: The yield column measures the effectiveness of the chemical reactions. Reaction yields are an essential metric for assessing the efficacy of the reaction conditions as well as a measure of the success of the reaction. 
