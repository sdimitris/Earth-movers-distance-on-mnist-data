import pulp as p
import numpy as np
import argparse
from math import *

class EMD:
    def __init__(self,input_file,query_file,datalbls,arg_querylbls,cluster_size):
        self.input_file = input_file
        self.query_file = query_file
        self.cluster_size = cluster_size
        self.clusters = 0
        self.suppliers = {}
        self.consumers = {}
        self.cols = 0
        self.rows = 0
        self.datalbls = datalbls
        self.arg_querylbls = arg_querylbls
    def readFiles(self):
        intType = np.dtype( 'int32' ).newbyteorder( '>' )
        nMetaDataBytes = 4 * intType.itemsize
        train = np.fromfile(self.input_file, dtype = 'ubyte' )
        magic, nImages, cols, rows = np.frombuffer( train[:nMetaDataBytes].tobytes(), intType )
        train = train[nMetaDataBytes:].astype( dtype = 'int32' ).reshape( [ nImages, cols, rows ] )

        queries = np.fromfile(self.query_file, dtype = 'ubyte' )
        magic, nImages, cols, rows = np.frombuffer( queries[:nMetaDataBytes].tobytes(), intType )
        queries = queries[nMetaDataBytes:].astype( dtype = 'int32' ).reshape( [ nImages, cols, rows ] )

        nMetaDataBytes = 2 * intType.itemsize
        trainlbls = np.fromfile(self.datalbls, dtype = 'ubyte' )
        magic, nTrainLabels = np.frombuffer( trainlbls[:nMetaDataBytes].tobytes(), intType )
        trainlbls= trainlbls[nMetaDataBytes:].astype( dtype = 'int32' )
        self.trainlbls = trainlbls

        querylbls = np.fromfile(self.arg_querylbls, dtype = 'ubyte' )
        magic, nTrainLabels = np.frombuffer( querylbls[:nMetaDataBytes].tobytes(), intType )
        querylbls= querylbls[nMetaDataBytes:].astype( dtype = 'int32' )
        self.querylbls = querylbls


        self.clusters = int(cols*rows/(self.cluster_size**2))
        self.train = train
        self.queries = queries
        self.cols = cols
        self.rows = rows
        self.costs = np.zeros((self.clusters,self.clusters),dtype="float")

    def makeClusters(self,image,option,image_index):
        if option == "train":
            self.suppliers[image_index] = []
        else:
            self.consumers[image_index] = []
        for i in range(0,int (sqrt(self.clusters))):
            for j in range(0,int (sqrt(self.clusters))):
                weight = 0 
                for x in range(0,self.cluster_size):
                    for y in range(0,self.cluster_size):
                        weight += image[(i*self.cluster_size)+x][(j*self.cluster_size)+y]
                        if (x == self.cluster_size//2 and y == self.cluster_size//2):
                            center = ((i*self.cluster_size)+x,(j*self.cluster_size)+y)
                if option == "train":
                    self.suppliers[image_index].append((weight,center))
                else:
                    self.consumers[image_index].append((weight,center))
        
    def makeCosts(self):
        i = 0
        j = 0
        for item1 in self.suppliers[0]:
            j = 0
            for  item2 in self.suppliers[1]:
                xi,xj = item1[1]
                yi,yj = item2[1]
                self.costs[i][j] = sqrt(abs(xi-yi) * abs(xi-yi) + abs(xj-yj)*abs(xj-yj))
                j+=1
            i += 1

    
def manh_distance(array1,array2):
    sum = 0
   
    for i in range(0,len(array1)):
        for j in range(0,len(array1[i])):
            sum = sum + abs(array1[i][j] - array2[i][j])
    return sum
        
                            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data")
    parser.add_argument("-q","--query")
    parser.add_argument("-s","--size")
    parser.add_argument("-l1","--datalbls")
    parser.add_argument("-l2","--querylbls")
    parser.add_argument("-o","--output")
    args = parser.parse_args()
    emd = EMD(args.data,args.query,args.datalbls,args.querylbls,int(args.size))
    emd.readFiles()
    for i in range(0,len(emd.queries)):
       emd.makeClusters(emd.queries[i],"queries",i)
    for i in range(0,len(emd.train)):
        emd.makeClusters(emd.train[i],"train",i)
    emd.makeCosts()
    ys = [p.LpVariable("y{}".format(i), cat="Binary") for i in range(0,emd.clusters)]
    supply = {}
    demand = {}
    average_emd = 0
    average_manh = 0
    print("Number of queries:",len(emd.queries))
    print("Number of train images",len(emd.train))
    print("Ready to operate!")
    for query_index in range (0,5):#len(emd.consumers)):
        print("Searching for Query ", query_index)
        neighbors_emd = []
        neighbors_manh = []
        i = 0
        for y in ys:
            demand[y] = emd.consumers[query_index][i][0]
            i+=1
        min_cost = float("inf")
        min_picture = 0
        min_cost_manh = float("inf")
        min_picture_manh = 0
        for index in range (0,100):#len(emd.suppliers)):
            emd_problem = p.LpProblem('emd',p.LpMinimize)
            xs = [p.LpVariable("x{}".format(i), cat="Binary") for i in range(0,emd.clusters)]
            i = 0
            for x in xs:
                supply[x] = emd.suppliers[index][i][0]
                i+=1
            routes = [(x,y) for x in xs for y in ys] # Creates a list of tuples containing all the possible routes for transport
            route_vars = p.LpVariable.dicts("Route",(xs,ys),0,None,p.LpContinuous)
            emd_problem += p.lpSum([route_vars[x][y]*emd.costs[int(str(x)[1:])][int(str(y)[1:])] for (x,y) in routes])
            for x in xs:
                emd_problem += p.lpSum([route_vars[x][y] for y in ys]) <= supply[x]
            for y in ys:
                emd_problem += p.lpSum([route_vars[x][y] for x in xs]) >= demand[y]
            emd_problem.writeLP("emd.lp")
            
            emd_problem.solve(p.PULP_CBC_CMD(msg=False))
            min_cost = p.value(emd_problem.objective)
            min_picture = index
            neighbors_emd.append((min_cost,min_picture))
            
            ########## calculate manhatan best neighbor ##########
            
            temp = manh_distance(emd.train[index],emd.queries[query_index])
            min_cost_manh = temp
            min_picture_manh = index
            neighbors_manh.append((min_cost_manh,min_picture_manh))

        neighbors_emd.sort(key=lambda tup: tup[0])  # sorts in place
        neighbors_manh.sort(key=lambda tup: tup[0])  # sorts in place
        correct_manh  = 0
        correct_emd = 0
        counter = 0
        for item in neighbors_emd[:10]:
            if emd.querylbls[query_index] == emd.trainlbls[item[1]]:
                correct_emd += 1
        counter = 0
        for item1 in neighbors_manh[:10]:
            if emd.querylbls[query_index] == emd.trainlbls[item1[1]]:
                correct_manh += 1
            counter += 1

        average_emd += correct_emd/10
        average_manh += correct_manh/10
    formatted_float_emd = "{:.2f}".format((average_emd/5)*100)#len(emd.queries))*100)
    formatted_float_manh = "{:.2f}".format((average_manh/5)*100)#len(emd.queries))*100)
    print("Average Correct Search Results EMD:",formatted_float_emd,"%")
    print("Average Correct Search Results MANHATTAN:",formatted_float_manh,"%")
    f = open(args.output, "a")
    f.write("Average Correct Search Results EMD: " + str(formatted_float_emd) + "%\n")
    f.write("Average Correct Search Results MANHATTAN: " + str(formatted_float_manh) + "%")
    f.close()
    


    
