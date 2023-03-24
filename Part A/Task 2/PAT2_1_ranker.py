import pickle
from posixpath import join
from queue import PriorityQueue
from math import log2,log10
import math
import time
import sys

inverted_index={} #store the inverted index
key_index_map={} #stores the term and term_id mapping

'''
This function iterates entire inverted index and stores dictionary in the form {doc_id:list of [term_id,tf]}.
Thus, it eliminates the need for rescanning entire inverted index or corpus to find term frequency to calculate cosine similarity.
'''
def getcorpus():
    corpus={}
    global inverted_index,key_index_map
    
    for i in inverted_index:
           
        for j in inverted_index[i]:
            if j[0] not in corpus:
                corpus[j[0]]=[]
            idx=key_index_map[i]
            corpus[j[0]].append((idx,j[1]))
    for i in corpus:
        corpus[i]=sorted(corpus[i])
    return corpus

'''
To read the inverted index stored in Task 1a.
Also, maps key of inverted index(which is term) to term_id
'''
def read_index(f):
    global inverted_index,key_index_map,one,zero
    inverted_index = pickle.load(f)
    key_list=list(inverted_index.keys())
    j=0
    # print("index: ",len(key_list))
    for i in key_list:
        key_index_map[i]=j
        j=j+1

# To compute Document Frequency
def doc_freq():
    df=[0]*len(inverted_index)
    global key_index_map
    for i in inverted_index:
        idx=key_index_map[i]
        df[idx]=len(inverted_index[i])
        
    return df

# Function to compute tf-idf value in the form of lnc:ltc
def lnc_ltc(doc_token, query_token, vec_df, N):
    c1 = 0
    c2 = 0
    ans = 0
    i = 0
    j = 0
    while i < len(doc_token) and j < len(query_token):
        if(doc_token[i][0] < query_token[j][0]):
            # tf-idf for document
            dtf = 1+log10(doc_token[i][1])
            c1 += dtf**2
            i = i+1
        elif(doc_token[i][0] > query_token[j][0]):
            # tf-idf for queries
            qtf =( (1+log10(query_token[j][1])) * log10(N/vec_df[query_token[j][0]]))
            c2 += qtf**2
            j = j+1
        else:
            # tf-idf for document
            dtf = 1+log10(doc_token[i][1])
            c1 += dtf**2
            # tf-idf for queries
            qtf = ((1+log10(query_token[j][1])) * log10(N/vec_df[query_token[j][0]]))
            c2 += qtf**2
            # Final tf-idf without normalization
            ans += dtf*qtf
            i = i+1
            j = j+1
    
    while i < len(doc_token):
        dtf = 1+log10(doc_token[i][1])
        c1 += dtf**2
        i = i+1
    
    while j < len(query_token):
        qtf = ((1+log10(query_token[j][1])) * log10(N/vec_df[query_token[j][0]]))
        c2 += qtf**2
        j = j+1

    # Final Output is returned, after applying cosine for normalization
    if c1 == 0 or c2 == 0:
        return 0
    else:
        return ans/(math.sqrt(c1*c2))

# Function to compute Lnc:Lpc
def Lnc_Lpc(doc_token,query_token,vec_df,N):
    c1=0
    c2=0
    ans=0
   
    prob_idf = 0
    avg_doc = 0
    avg_query = 0
    
    # Avg of the term frequency for document is calculated
    for i in range(len(doc_token)):
        avg_doc += doc_token[i][1]
    
    # Denominator is computed for Document of L (log Average)
    avg_doc = avg_doc / len(doc_token)
    avg_doc = log10(avg_doc)
    avg_doc = 1 + avg_doc

    # Avg of the term frequency for queries is calculated
    for i in range(len(query_token)):
        avg_query += query_token[i][1]
    
    # Denominator is computed for queries of L (log Average)
    avg_query = avg_query / len(query_token)
    avg_query = log10(avg_query)
    avg_query = 1 + avg_query
    
    i=0
    j=0

    while i<len(doc_token) and j<len(query_token):
        if(doc_token[i][0]<query_token[j][0]):
            dtf=1+log10(doc_token[i][1])
            dtf = dtf/avg_doc
            c1+=dtf**2    
            i=i+1
        elif(doc_token[i][0]>query_token[j][0]):
            # prob idf is computed for queries
            prob_idf = max(0, (log10((N-vec_df[query_token[j][0]])/vec_df[query_token[j][0]])))
            qtf=(((1+log10(query_token[j][1]))/avg_query) * prob_idf)
            c2+=qtf**2
            j=j+1
        else:
            dtf=1+log10(doc_token[i][1])
            dtf=dtf/avg_doc
            c1+=dtf**2 
            # prob idf is computed for queries   
            prob_idf = max(0, (log10((N-vec_df[query_token[j][0]])/vec_df[query_token[j][0]])))
            qtf=(((1+log10(query_token[j][1]))/avg_query) * prob_idf)
            c2+=qtf**2
            ans+=dtf*qtf 
            i=i+1
            j=j+1
    
    while i < len(doc_token):    
        dtf=1+log10(doc_token[i][1])
        dtf = dtf/avg_doc
        c1+=dtf**2    
        i=i+1
    
    while j < len(query_token):
        # prob idf is computed for queries
        prob_idf = max(0, (log10((N-vec_df[query_token[j][0]])/vec_df[query_token[j][0]])))
        qtf=(((1+log10(query_token[j][1]))/avg_query) * prob_idf)
        c2+=qtf**2
        j=j+1

    # Final Output is returned, after applying cosine for normalization
    if c1==0 or c2==0:
        return 0
    else:
        return ans/(math.sqrt(c1)*math.sqrt(c2))

# Function to compute anc:apc
def anc_apc(doc_token,query_token,vec_df,N):
    c1=0
    c2=0
    ans=0
      
    prob_idf = 0
    max_doc = -1
    max_query = -1
    
    # Max of the term frequency for document is calculated
    for i in range(len(doc_token)):
        if(doc_token[i][1] > max_doc):
            max_doc = doc_token[i][1]
    
    # Max of the term frequency for queries is calculated
    for i in range(len(query_token)):
        if(query_token[i][1] > max_query):
            max_query = query_token[i][1]
    
    i=0
    j=0
    
    while i<len(doc_token) and j<len(query_token):
        if(doc_token[i][0]<query_token[j][0]):
            dtf= 0.5+(0.5*doc_token[i][1]/max_doc)
            c1+=dtf**2    
            i=i+1
        elif(doc_token[i][0]>query_token[j][0]):
            # prob idf is computed for queries
            prob_idf = max(0, (log10((N-vec_df[query_token[j][0]])/vec_df[query_token[j][0]])))
            qtf=((0.5+(0.5*query_token[j][1]/max_query)) * prob_idf)
            c2+=qtf**2
            j=j+1
        else:
            dtf= 0.5+(0.5*doc_token[i][1]/max_doc)
            c1+=dtf**2 
            # prob idf is computed for queries  
            prob_idf = max(0, (log10((N-vec_df[query_token[j][0]])/vec_df[query_token[j][0]])))
            qtf=((0.5+(0.5*query_token[j][1]/max_query)) * prob_idf)
            c2+=qtf**2
            ans+=dtf*qtf 
            i=i+1
            j=j+1
    
    while i < len(doc_token):    
        dtf= 0.5+(0.5*doc_token[i][1]/max_doc)
        c1+=dtf**2    
        i=i+1
    
    while j < len(query_token):
        # prob idf is computed for queries
        prob_idf = max(0, (log10((N-vec_df[query_token[j][0]])/vec_df[query_token[j][0]])))
        qtf=((0.5+(0.5*query_token[j][1]/max_query)) * prob_idf)
        c2+=qtf**2
        j=j+1
    
    # Final Output is returned, after applying cosine for normalization
    if c1==0 or c2==0:
        return 0
    else:
        return ans/(math.sqrt(c1)*math.sqrt(c2))

def store_list(file,lis):
    write_file=open(file,'w')
    for i in lis:
        op=i[0]+":"+i[1]+"\n"
        write_file.write(op)
    
    write_file.close()

# queries (output of task-1B) are tokenaized 
def query_tokens(query):
    global key_index_map
    terms=query.split()
    freq={}
    for i in terms:
        
        if i not in inverted_index:
            continue
        idx=key_index_map[i]
        if idx in freq:
            freq[idx]=freq[idx]+1
        else:
            freq[idx]=1
    return list(sorted(freq.items())) 

# main function
def main():
    t1=time.time()
    model_queries = sys.argv[1]
    queries = sys.argv[2]

    # Reading model queries file (output of Task-1A)
    index_file=open(model_queries,"rb")

    # Reading queries file (output of Task-1B)
    read_file = open(queries, "r")
    read_index(index_file)
    t2=time.time()
    print("reading index ",t2-t1)
    df=doc_freq()
    
    corpus=getcorpus()
    total_docs=len(corpus)
    # print("corpus ",total_docs)
    t3=time.time()
    print("reading corpus ",t3-t2)

    # 3 files are initialized
    file1=open('PAT2_1_ranked_list_A.csv','w')
    file2=open('PAT2_1_ranked_list_B.csv','w')
    file3=open('PAT2_1_ranked_list_C.csv','w')

    file1.write('Query_ID,Document_ID\n')
    file2.write('Query_ID,Document_ID\n')
    file3.write('Query_ID,Document_ID\n')
    
    query_list=[]
    for line in read_file:
        q=line.split(',')
        query_list.append((q[0],query_tokens(q[1][:-1])))
    
    for line in query_list:
        l1=[]
        l2=[]
        l3=[]
        print(line[0])
                
        for i in corpus:
            # tf-idf is computed by using all the 3 methods
            s1=lnc_ltc(corpus[i],line[1],df,total_docs)
            s2=Lnc_Lpc(corpus[i],line[1],df,total_docs) 
            s3=anc_apc(corpus[i],line[1],df,total_docs)
            l1.append((s1,i))
            l2.append((s2,i))
            l3.append((s3,i))  
        l1.sort(reverse=True)
        l2.sort(reverse=True)
        l3.sort(reverse=True)
        
        # Top 50 document names are ranked in the form of (query_id,document_id)
        for j in range(0,50):    
            file1.write(str(line[0]+','+l1[j][1]+'\n'))
            file2.write(str(line[0]+','+l2[j][1]+'\n'))
            file3.write(str(line[0]+','+l3[j][1]+'\n'))
    
    # Total time to score all the documents 
    t6=time.time()
    print("document scoring time ",t6-t3)
    
    read_file.close()
    file1.close()   
    file2.close()
    file3.close()
    index_file.close()

if __name__ == "__main__":
    main()
