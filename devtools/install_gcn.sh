# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# temporarily change directory to install software
pushd .
cd ..

# clone the repo in the parent directory and update it
echo "Cloning/Updating GCN..."
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
cd TS-GCN || exit
git fetch origin
git checkout main
git pull origin main

# Add to PYTHONPATH
echo "Adding GCN to PYTHONPATH..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo 'export PYTHONPATH=$PYTHONPATH:'"$(pwd)" >> ~/.bashrc
echo $PYTHONPATH

# create the environment
echo "Creating the GCN environment..."
source ~/.bashrc
make conda_env

# Restore the original directory
cd ../ARC || exit
echo "Done installing GCN."
popd || exit
