# Example models

## Test ONNX models

The models defined in `test.py` are used for tests. Run `python test.py`
to create correspoding onnx-files in folder `test` which can be viewed with [netron](netron.app).

## Multi-Layer-Perceptron

Run `python mlp.py` to create and train a very simple MLP classifying the
[circles dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html).
and store it to `mlp.onnx`.


## Vision Transformer for Image Classification

Run `python vit_image_classifier.py` to download a trained
[ViT Base](https://huggingface.co/google/vit-base-patch32-224-in21k)
from the huggingface model repository and store it to `vit_image_classifier.onnx`.