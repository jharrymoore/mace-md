#!/bin/bash

srun -n 12 --gpu-bind=map_gpu:0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 mace-md -f /path/to/input_file.pdb\
  --ml_mol /path/to/input_file.pdb \
  --resname "LIG"\
  --output_dir "./16_reps_production" \
  --pressure 1.0 \
  --decouple \
  --restart \
  --replicas 12 \
  --log_level DEBUG \
  --nl "nnpops" \
  --run_type "repex" \
  --system_type "pure" \
  --dtype float64 \
  --interval 1 \
  --steps 5000 \
  --model_path "/path/to/MACE.model" \
  --steps_per_iter 1000 \
