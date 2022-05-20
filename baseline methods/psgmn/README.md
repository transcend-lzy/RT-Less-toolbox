# PSGMN


We used the psgmn was presented in "Pseudo-Siamese Graph Matching Network for Textureless Objects' 6D Pose Estimation" .

## Installation

1. Set up the python environment:
    ```
    conda create -n psgmn python=3.7
    conda activate psgmn
    ```
    ### install torch 1.5 built for cuda 10.1
    ```
    conda install pytorch==1.5.0 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
    ```
    ### install [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
    ```
    pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-geometric
    ```
    ### install other requirements
    ```
    pip install -r requirements.txt
    ```
    ### compile the cuda extension
    ```
    cd csrc
    python setup.py build_ext --inplace 
    ```
## Training
Take the training on `obj1` as an example.
   run

   ```
   python main_psgmn.py --class_type obj1 --train True
   ```
## Testing

### Testing on Linemod

Take the testing on `obj1` as an example.


1. Train model of `obj1` and put it to `$ROOT/model/obj1/150.pth`.
2. Test:
    ```
    python main_psgmn.py --class_type obj1 --eval True
    ```

# References

------

[1]C. Wu, L. Chen, Z. He, and J. Jiang, “Pseudo-Siamese Graph Matching Network for Textureless Objects’ 6D Pose Estimation,” *IEEE Trans. Ind. Electron.*, pp. 1–1, 2021, doi: 10.1109/TIE.2021.3070501.

[psgmn](https://github.com/Ray0089/PSGMN)
