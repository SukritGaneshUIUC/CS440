One time only:
conda create --name cs440-mp5 python=3.7 numpy

###############################################################

conda activate cs440-mp5
cd Documents/UIUC Material/CS440/MP5/mp5-code

python mp5.py --dataset data/mp5_data --method perceptron
python mp5.py --dataset data/mp5_data --method knn
