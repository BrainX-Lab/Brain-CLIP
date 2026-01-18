# Brain-CLIP Documentation

## Usage
1. Specify the path and file names to the SC/FC data in [utils.py](utils.py).
2. Specify name, number of trials, max epochs, input data size (`N_ROI`), and batch size in [batch_train.py](batch_train.py).
3. Optional: additional parameters for training used in [batch_train.py](batch_train.py).
    * `k`: the exponent on the ground truth similarity matrix used for loss calculation.
        - Higher values will emphasize differences between individuals.
    * `alpha`: the trade-off between losses for cross-modality similarity vs. same-modality similarity. 
        - Higher values will prioritize retaining relative similarities between individuals embeddings for the same mode.
        - Lower values will prioritize encoding similar embeddings between the different modes for the same individual.
    * `beta`: the trade-off between losses for Brain-CLIP and regeneration/prediction
        - Higher values will prioritize the accuracy of regenerating a person's SC from their FC embedding and vice versa.
        - Lower values will prioritize encoding embeddings that are similar across different modes for the same individual and retain relative differences between individuals for the same mode.
4. Run [batch_train.py](batch_train.py).

## Files
* [`batch_train.py`](batch_train.py) contains the training loop for the model.
* [`train.py`](train.py) contains functions that are used for training/testing at each individual epoch.
* [`model.py`](model.py) contains the model architecture.
* [`BNT_modules`](BNT_modules.py) contains the transformer encoders used in the model.
    - Part of the source code comes from [https://github.com/Wayfear/BrainNetworkTransformer](https://github.com/Wayfear/BrainNetworkTransformer), used for accessing the attention weights of transformer layers.
* [`utils.py`](utils.py) contains:
    - The path to the SC/FC data. 
    - Helper functions for data loading/preprocessing
    - Functions for calculating cosine similarities and PCC loss (part of CPL loss in the paper)