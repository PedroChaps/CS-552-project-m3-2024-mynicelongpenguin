# /bin/bash

module load gcc python cuda

virtualenv --system-site-packages ~/venvs/venv_evaluator

source ~/venvs/venv_evaluator/bin/activate


pip install --upgrade pip

pip3 install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r ./requirements.txt
pip3 install -U git+https://github.com/huggingface/trl

deactivate

