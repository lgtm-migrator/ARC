# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -n tani_env python=3.7 -y
conda activate tani_env
conda install -c conda-forge torchani qcelemental -c anaconda yaml -y
conda deactivate
