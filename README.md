# Earth-mover's-distance-on-mnist-data
Earth mover's distance implementation on mnist data using the pulp library of python for solving linear transportation problems

##
Because the data is very heavy and the execution time is very long , this script performs only 5 queries over 100 images( train data set).
Αlso, this script finds the 10 nearest neighbors of each query  and calculates the accuracy comparing the train/query labels.

You need to download the pulp,numpy libraries and the mnist data to run the script.
# Execution instrtuction
python search.py -d train-images.idx3-ubyte -q t10k-images.idx3-ubyte -s "size of subcluster" -l1 train-labels.idx1-ubyte -l2 t10k-labels-idx1-ubyte
