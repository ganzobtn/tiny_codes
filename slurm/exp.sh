#!/bin/bash
#SBATCH --job-name=run.sh             # Job name

#SBATCH --time=1:00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --output=/home/ganzorig.batnasan/tiny_codes/multinodes_slurm/%x-%j.out

#hostname

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


source /apps/local/conda_init.sh
conda activate vit

### the command to run
srun python main.py --net resnet18 \
--lr 1e-3 --epochs 50 --other_args