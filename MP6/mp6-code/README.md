One time only:
conda create --name cs440-mp6 python=3.7 numpy pytorch

###############################################################

conda activate cs440-mp6
cd Documents/UIUC Material/CS440/MP6/mp6-code

python mp6.py -h
python mp6.py --dataset data/mp6_data --max_iter 500 --part 1
python mp6.py --dataset data/mp6_data --max_iter 500 --part 2

conda deactivate
