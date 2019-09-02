conda create --name vico_env
conda activate vico_env
conda install -y -c anaconda cudatoolkit=8.0
conda install -y -c pytorch pytorch=0.3.1 cuda80
conda install -y -c pytorch torchvision=0.2.0
conda install -y -c anaconda nltk ujson h5py scipy pandas scikit-learn scikit-image pyyaml
conda install -y -c plotly plotly
conda install -y -c conda-forge tqdm tensorboard
conda install -y h5py
pip install tensorboard_logger