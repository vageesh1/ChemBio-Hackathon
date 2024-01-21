# ChemBio-Hackathon
This is our submission for the ChemBio AI hackathon conducted by ChemBio AI under Prometeo'24 <br>
Report- [Report Link](https://github.com/vageesh1/ChemBio-Hackathon/blob/main/Report.pdf)<br>
Slides- [Slides Link](https://github.com/vageesh1/ChemBio-Hackathon/blob/main/ChemBio%20Hackathon.pdf)

## Problem Statement
Build a seq to seq based model which could predict the reaction conditions for the reaction i.e. given reaction smiles as an input it should be able to predict the conditions.

## Problem Description 
To give an overview of the problem statement, We are given a dataset that contains the organic reaction in smiley format. The conditions on which the prediction is required are Reagent, Solvent, and Catalyst based on the Given Organic reaction

## Data Description 
1. **canonic_rxn**: This column contains the canonical representation of chemical reactions using SMILES (Simplified Molecular Input Line Entry System) strings. Each row represents a distinct chemical transformation, capturing the reactants and products involved. 
2. **rxnmapper_aam**: This column encodes atom-to-atom mappings (AAM) for the reaction. AAM is a technique used to establish correspondence between atoms in reactants and products, facilitating the tracking of atom transformations during a reaction. 
3. **Reagent**: This column details the reagents utilized in the chemical reactions. Reagents are chemical substances introduced into a reaction to initiate or facilitate the transformation. 
4. **Solvent**: The solvents used are listed in the solvent column for each reaction. Reaction conditions are greatly influenced by solvents, which also have an impact on reaction rates and results. 
5. **Catalyst**: Substances known as catalysts quicken reactions without changing permanently. Understanding catalysts helps one gain an understanding of the efficiency and mechanism of reactions. 
6. **Yield**: The yield column measures the effectiveness of the chemical reactions. Reaction yields are an essential metric for assessing the efficacy of the reaction conditions and measuring the success of the reaction. 
![Dataset](https://github.com/vageesh1/BioChem-Hackathon/blob/main/Dataset.jpg)


## Methodology 
For this we used a graph-based seq2seq network. With a graph fused encoder-based encoder and the transformer decoder for the decoder. The input goes for the smiley string and the output comes out as all three different Reagants.
Here is an overview of Model Architectures
1. Any smiles contains a reaction, so first task is to separate both the reactant and product.
2. Each of the product and reactant are parsed through the Graph Encoder which first gets the graph features and then apply an attention encoder to it
3. Then A decoder is attached and then the output of both is fused into a single space which is projected through 3 different classes to give our 3 different outputs
4. The tokenizing and padding are based on count vectorizer and spacy
5. The output comes as a 1d tensor which can be later decoded to be exact outputs

## Model Archietecure 
Here is an overview of the architecture and the flow of variables in our architecture. 
Given below is the flow of variables, how the flow of variables is happening, what are the inputs and what are their respective outputs
![Flow of Variables](https://github.com/vageesh1/BioChem-Hackathon/blob/main/Flow%20of%20Variables.jpg)<br>
Given Below is the Detailed architecture 
![Architecture](https://github.com/vageesh1/BioChem-Hackathon/blob/main/architecture.jpg)<br>


## Training Details
For Training, I have used Adam as the Optimizer and the MSE for the loss calculation, for each output loss is calculated and total loss is calculated by adding all the different loss and backpropagating the loss
I trained it for 10 epochs, here is the loss curve
With the ongoing epochs, the loss is decreasing with each epoch, showing a combined use loss of 300 range which keeps decreasing with each epoch
![Loss Curve](https://github.com/vageesh1/BioChem-Hackathon/blob/main/Loss.jpg)


## Inference 
For inferencing, I have initialized the target sequences as zeros and the output probability is being calculated which I am later decoding using the dictionary I made from tokenizer by getting the ceratin string corresponding to the output 
![The Inference Result](https://github.com/vageesh1/BioChem-Hackathon/blob/main/Inference%20Result.jpg)

## Files Descriptions 
**Notebooks**- This folder contains all the experiments, preprocessing, and sample architectures, The **full Training Loop.ipynb** is our main file in which we compiled all of the architectures, pipelines and compiled into one and did the training. 
Rest are the python files for each of the different component present, to do the training if needed on local machine 


