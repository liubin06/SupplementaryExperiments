# Supplementary experiments for BNS with different loss functions.

- By fixing the loss function as BPR loss, MF  and lightGCN have shown that  BNS is applicable to **Different Encoders**. Implement BNS for Matrix Factorization (MF) (run [main_MF.py](https://github.com/liubin06/BNS/blob/main/BNS_MF/main_MF.py) ); 
Implement BNS for light Graph Convolution Network (LightGCN) (run [main_lightGCN.py](https://github.com/liubin06/BNS/blob/main/BNS_lightGCN/main_lightGCN.py) ).

- By fixing the encoder as MF, this supplementary experiments will show that BNS is applicable to **Different Loss Functions**.




## Loss Functions
- BCE  : Implement BNS for binary corss-entropy loss (run [MF_BCE.py](https://github.com/liubin06/SupplementaryExperiments/blob/main/MF_BCE.py) ); 
- InfoNCE: Implement BNS for InfoNCE loss(run [MF_InfoNCE.py](https://github.com/liubin06/SupplementaryExperiments/blob/main/MF_InfoNCE.py ) ).

## Why BNS is applicable to contrastive-based loss functions?

The BCE loss, BPR loss and Info NCE loss encourages the model to score positive instances higher than negative instances by pulling learned feature representation of positive instance to be similar with "anchor" data point, while pushing features of negative instance apart from "anchor" data point in the embedding space. This implies the order relation also might hold in general for contrastive-based loss functions.

BNS derived from the order relation, leading it applicable to other contrastive based loss functions where the order relation holds, such as BCE loss, Info NCE loss.

## Gradient descent for different loss functions
### BCE loss
Draw positive instance (u,i) or negative instance (u,j). 

- If $x \in pos$, updating $\mathbf{u}$ and $\mathbf{i}$ by performing: 

$\mathbf{u} \leftarrow \mathbf{u} + \alpha \times [1-\sigma(\hat{x}_{ui})]\times \mathbf{i}$, and 

$\mathbf{i} \leftarrow \mathbf{i} + \alpha \times [1-\sigma(\hat{x}_{ui})]\times \mathbf{u}$.

- If $x \in neg$, updating $\mathbf{u}$ and $\mathbf{j}$ by performing: 

$\mathbf{u} \leftarrow \mathbf{u} + \alpha \times \sigma(\hat{x}_{uj})\times (-\mathbf{j})$, and 

$\mathbf{j} \leftarrow \mathbf{j} + \alpha \times \sigma(\hat{x}_{uj}) \times (-\mathbf{u})$.

### InfoNCE loss with one negative 
Draw (u,i,j), updating $\mathbf{u}$, $\mathbf{i}$ and $\mathbf{j}$ by performing: 

$\mathbf{u} \leftarrow \mathbf{u} + \alpha \times (1-P) \times (\mathbf{i}-\mathbf{j})$, 

$\mathbf{i} \leftarrow \mathbf{i} + \alpha \times (1-P) \times \mathbf{u}$,

$\mathbf{j} \leftarrow \mathbf{j} + \alpha \times (1-P) \times (-\mathbf{u})$ , where 

$P = \frac{e^{\hat{x}_{ui}}}{e^{\hat{x}_{ui}} + e^{\hat{x}_{uj}}}$

## Experimental results
<div align=center>
<img src="https://github.com/liubin06/SupplementaryExperiments/blob/main/Results.png">
</div>
