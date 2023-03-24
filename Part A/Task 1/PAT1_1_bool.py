import pickle
from queue import PriorityQueue
import time
import sys

inverted_index = None #To store an Inverted Index

'''
This function find an intersection between two posting list.
Input: posting list in an ascending order of document id
Output: List of document ids which are present in both the posting list
'''
def merging_lists(list_a, list_b):
    i = 0
    j = 0
    list_c = []
    while (i < len(list_a) and j < len(list_b)):
        if(list_a[i][0] > list_b[j][0]):
            j = j+1
        elif (list_a[i][0] < list_b[j][0]):
            i = i+1
        else:
            list_c.append(list_a[i])
            i = i+1
            j = j+1
    return list_c

#Creating a priority queue to store posting list along with it's priority
def fetch_all_query_terms(q):
    term_queue = PriorityQueue()
    temp_list = q.split()
    
    for i in temp_list:
        if i in inverted_index:
            size = len(inverted_index[i])
            term_queue.put((size, inverted_index[i])) #size act as an priority.
    return term_queue

#reading inverted index file stored in task 1A
def read_index():
    global inverted_index
    with open('model_queries_1.pth', 'rb') as f:
        inverted_index = pickle.load(f)


def main():
    read_index()

    model_queries = sys.argv[1]
    queries = sys.argv[2]

    # Reading model queries file (output of Task-1A)
    index_file=open(model_queries,"rb")

    # Reading queries file (output of Task-1B)
    read_file = open(queries, "r")
    
    write_file = open("PAT1_1_results.txt", "w")  #storing the boolean retrieval result in query_id:list of docs format
    
    for line in read_file:
        query = line.split(',')
        terms = fetch_all_query_terms(query[1][:-1])  #store posting list for each term in query along with it's priority
        op = query[0]+' : '
        if terms.qsize() == 0:
            op += '\n'
            write_file.write(op)
            continue
        '''
        Select two smallest posting list from priority queue and find intersection.
        Store the resultant list into priority queue.
        Repeat this process untill there is only one list remains in the queue.
        '''
        while terms.qsize() != 1:
            a, lista = terms.get()
            b, listb = terms.get()
        
            listc = merging_lists(lista, listb)
            terms.put((len(listc), listc))

        l, list_op = terms.get()
        for i in range(0, l):
            op += list_op[i][0]
            if(i < l-1):
                op += ' '
        op += '\n'
        write_file.write(op)

    read_file.close()
    write_file.close()
    print("Done")

if __name__ == "__main__":
    main()