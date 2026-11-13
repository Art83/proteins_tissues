# Gene by tissue interaction 
This project will attempt to solve the remaining question from hpa_to_tabula atlas: How much the protein expression is defined by interaction of context (cell types/tissue) and gene identity.


## Results
**script 0 and 1** are setting the stage and put together the dataset. NO results there

**script 2** built and additive logistic model to set the baseline. logit = μ + b(g) + d(c). AUC within-gene 0.74

**script 3** adds an interaction term and determines the optimal number of latent factors in nn.Embeddings. logit = μ + b(g) + d(c) + u(g) · v(c). AUC within gene grew to 0.83 which is 11% gain attributed to gene*context interaction

**script 3b** explains what drives the model in script 03 and why the fancy model is essentially not needed. We have comparable AUC just using similarity of the contexts