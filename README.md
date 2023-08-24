# A^2MR-GNN
The A^2MR-GNN defines the prediction of DTBA as a regression task, in which the model’s input is the drug-target representation, and the output is a continuous value representing the binding affinity score between the drug and the target protein. The overall architecture of the A^2MR-GNN is shown in the figure below.
![image](https://github.com/ILangXu/A2MR-GNN/assets/37317304/39d6cd92-69a0-4bfd-acc1-9d0b9abf311c)

# Get Start


Necessary packages should be installed to run the A^2MR-GNN model.
1. Dependecies:
  
* python >= 3.7,  
* Pytorch (>=1.6.0),  
* numpy,
* scipy,
* scikit-learn.  
2. Datasets:  
We adopt the PDBbind dataset v2016 for experiments and employ two test sets (the v2016 and v2013 core sets) to test the performance of A^2MR-GNN.
Train the model:
3. Use the train.py script to train the model.
