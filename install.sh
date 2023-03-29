python3.8 -m venv .venv-dave2
. .venv-dave2/bin/activate
pip3 install torch==1.9.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
pip3 install numpy black mypy matplotlib scipy scikit-image pandas opencv-python
