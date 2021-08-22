One time only:
conda create --name cs440-mp4 python=3.7

###############################################################

conda activate cs440-mp4
cd Documents/UIUC Material/CS440/MP4/starter_code

python mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm baseline
python mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_1
python mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_2
python mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_3
