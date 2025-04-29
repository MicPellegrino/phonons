clear
units		metal
dimension	3
boundary	p p p
atom_style	atomic
atom_modify map yes
newton 		on

read_data 	W_bcc_unit.lmp
pair_style	mace no_domain_decomposition

# Here, specify the path to the .pt file for the NN potential:
# pair_coeff  * * <path-to-nnp> <element(s)>
pair_coeff  * * /home/michele/benchmark-mace/models-uob/GPU_Models/BCC/mace01_64channel_stagetwo.model-lammps.pt W
