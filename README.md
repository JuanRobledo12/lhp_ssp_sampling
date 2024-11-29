## Get Started

Create a conda env with python 3.11 (You can use any name)

```
conda create -n sisepuede python=3.11
```
Activate the env
```
conda activate sisepuede
```
Install the working version of the sisepuede package
```
pip install git+https://github.com/jcsyme/sisepuede.git@working_version
```
Install additional libraries
```
pip install -r requirements.txt
```

## How to Run

To generate simulations

```
cd src
bash run_sim_batch.sh {initial_sim_id} {final_sim_id} {yaml_file_name}
# Example
bash run_sim_batch.sh 0 100 config_croatia_batch_2.yaml
```

To group inputs and outputs
```
cd src
python3 reune_salidas.py {country_name} {batch_id}
# Example
python3 reune_salidas.py croatia 1
```
