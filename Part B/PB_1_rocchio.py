import   pickle
from math import log2, log10
import sys
import math
import pandas as pd
import time

# run command "python PB_1_rocchio.py model_queries_1.pth queries_1.txt PAT2_1_ranked_list_A.csv ./Data/rankedRelevantDocList.xlsx"

inverted_index = {} #store the inverted index created in PART 1
key_index_map = {} #store the term-id correspnding to term

#retrive provided golden standard ranking
def parseRelevantDocsList():
    '''
    It reads parseQueryDoc.xlxs document, create a dictionary of query ids along with their relevant document list and respective relevance scores
    '''
    file_name = sys.argv[4]
    
    sheet_name = 'RelevantDocs'

    # read RelevantDocs sheet name from input excel file
    df = pd.read_excel(file_name, sheet_name=sheet_name)

    # query ids and relevance scores are typecasted to numeric
    df['Query_ID'] = df['Query_ID'].astype(int)
    df['Relevance_Score'] = df['Relevance_Score'].astype(float)

    # create a dictionary to store the query ids along with their relevant document list and respective relevance scores
    query_document_rankings = dict()

    for i in range(len(df)):
        query_id = df.loc[i, 'Query_ID']
        document_id = df.loc[i, 'Document_ID']
        relevance_score = df.loc[i, 'Relevance_Score']

        if query_id not in query_document_rankings:
            query_document_rankings[query_id] = dict()

        query_document_rankings[query_id][document_id] = relevance_score

    return query_document_rankings


#retrive top 20 documents from ranking list PAT2_1_ranked_list_A.csv
def parseRankedListDoc():
    '''
    It reads list PAT2_1_ranked_list_A.csv and create dictionary of query ids along with their relevant document list
    '''
    ranked_list = sys.argv[3]
    df = pd.read_csv(ranked_list)

    # create dictionary to store query ids along with their relevant document list
    retrieved_rankings = dict()

    for i in range(len(df)):
        query_id = df.loc[i, 'Query_ID']
        document_id = df.loc[i, 'Document_ID']

        if query_id not in retrieved_rankings:
            retrieved_rankings[query_id] = []

        # only store top 20 relevant documents
        if len(retrieved_rankings[query_id]) < 20:
            retrieved_rankings[query_id].append(document_id)

    return retrieved_rankings

#merge two list. if both list have same term then sum up their values
def merging_lists(list_a, list_b):
    i = 0
    j = 0
    list_c = []
    while (i < len(list_a) and j < len(list_b)):
        if(list_a[i][0] > list_b[j][0]):
            list_c.append(list_b[j])
            j = j+1
        elif (list_a[i][0] < list_b[j][0]):
            list_c.append(list_a[i])
            i = i+1
        else:
            list_c.append((list_a[i][0],list_a[i][1]+list_b[j][1]))
            i = i+1
            j = j+1
    
    while (i<len(list_a)):
        list_c.append(list_a[i])
        i = i+1
    while (j<len(list_b)):
        list_c.append(list_b[j])
        j = j+1

    return list_c
    


#read term frequency of each term for each document
def getcorpus():
    corpus = {}
    global inverted_index, key_index_map

    for i in inverted_index:

        for j in inverted_index[i]:
            if j[0] not in corpus:
                corpus[j[0]] = []
            idx = key_index_map[i]
            corpus[j[0]].append((idx, j[1]))
    for i in corpus:
        corpus[i] = sorted(corpus[i])
    return corpus

#read inverted index and map term to term_id
def read_index(f):
    global inverted_index, key_index_map
    inverted_index = pickle.load(f)
    key_list = list(inverted_index.keys())
    j = 0
    for i in key_list:
        key_index_map[i] = j
        j = j+1

#convert query into list of tuple (term_id,frequency of term)
def query_tokens(query):
    global key_index_map
    terms = query.split()
    freq = {}
    for i in terms:

        if i not in inverted_index:
            continue
        idx = key_index_map[i]
        if idx in freq:
            freq[idx] = freq[idx]+1
        else:
            freq[idx] = 1
    return list(sorted(freq.items()))


#update the query using rochio's algorithm
def update_query(query,rel_docs,beta,non_rel_docs,gamma):
    updated_query = query
    
    if len(rel_docs)>0:
        temp_rel_docs=[]
        for j in rel_docs:
            temp_rel_docs.append((j[0],j[1]*beta))
        updated_query=merging_lists(updated_query,temp_rel_docs)

    if len (non_rel_docs)>0:
        temp_non_rel_docs=[]
        for j in non_rel_docs:
            temp_non_rel_docs.append((j[0],(-1)*j[1]*gamma))
        updated_query=merging_lists(updated_query,temp_non_rel_docs)
    
    return updated_query

#remove those term whose value become negative after updating the query using rochio's algorithm
def remove_negative(query_list):
    final_query=[]
    for i in query_list:
        if i[1]>0:
            final_query.append(i)
    return final_query


#get document frequency of each term
def doc_freq():
    df=[0]*len(inverted_index)
    global key_index_map
    for i in inverted_index:
        idx=key_index_map[i]
        df[idx]=len(inverted_index[i])
        
    return df

#lnc for document and ltc for query is pre-calculated to improve performance. And here only take dot product of the common words
def lnc_ltc(doc_token,cosine_doc,query_token,cosine_query):
    if(cosine_doc==0 or cosine_query==0):
        return 0
    i=0
    j=0
    ans=0
    for k in doc_token.keys()&query_token.keys():
        ans+=doc_token[k]*query_token[k]
    return ans/math.sqrt(cosine_doc*cosine_query) 

#return mAP and avg ndcg score for all queries
def evaluate(retrieved_rankings,query_document_rankings):
    map_20 = 0
    aver_ndcg_20 = 0
    no_of_queries = len(retrieved_rankings)

    for query_id in retrieved_rankings:
        # variables to store the average presision and ndcg
        ap_20 = 0
        ndcg_20 = 0
        no_of_relevant_20 = 0
        
        # store the relevance scores for top 10 as well as top 20 documents
        top_20_relevance = []
        ideal_top_20_relevance = []
        
        # retrieve top 20 among 50 relevant documnets from output file of PAT_1_ranker.py
        top_20 = retrieved_rankings[query_id]
        
        # if relevant document list for a particular query id  is not given 
        if query_id not in query_document_rankings:
            query_document_rankings[query_id] = dict()
        
        for i in range(len(top_20)):
            document_id = top_20[i]
            
            # if current document id is present in the relevant document list then retrieve its relevance score 
            if document_id in query_document_rankings[query_id]:
                relevance_score = query_document_rankings[query_id][document_id]
                
                
                # calculating metrics for top 20 relevant docs
                no_of_relevant_20 += 1
                ap_20 += no_of_relevant_20/(i+1)
                top_20_relevance.append(relevance_score)
                ideal_top_20_relevance.append(relevance_score)
            
            # if current document id is not present in the relevant document list then relevance score is set to 0
            # non relevant documents are not considered in average precision
            else:
                
                # calculating metrics for top 20 relevant docs
                top_20_relevance.append(0)
                ideal_top_20_relevance.append(0)
        
        # compute average precision
        if no_of_relevant_20 != 0:
            ap_20 = round(ap_20 / no_of_relevant_20, 4)
        
        map_20 += ap_20
        
        # ideal relevance score = sorted relevance scores in descending order
        ideal_top_20_relevance.sort(reverse=True)
        
        # compute the discounted cumulative gain for top 20 relevant docs   
        for i in range(1, len(top_20_relevance)):
            top_20_relevance[i] = top_20_relevance[i-1] + (top_20_relevance[i] / log2(i+1))
            ideal_top_20_relevance[i] = ideal_top_20_relevance[i-1] + (ideal_top_20_relevance[i] / log2(i+1))

        # compute the normalized discounted cumulative gain 
        if ideal_top_20_relevance[19] != 0:
            ndcg_20 = round(top_20_relevance[19] / ideal_top_20_relevance[19], 4)
        
        aver_ndcg_20 += ndcg_20
        
        
    # compute mean average precision    
    map_20 = round(map_20 / no_of_queries, 4)

    # compute average ndcg
    aver_ndcg_20 = round(aver_ndcg_20 / no_of_queries, 4)
    return map_20,aver_ndcg_20

#apply lnc scheme to all the documents
def tfidf_doc_vector(corpus):
    tfidf_doc={}
    cosine_doc={}
    for i in corpus:
        tfidf_doc[i]={}
        c1=0
        for j in corpus[i]:
            tfidf_doc[i][j[0]]=1+log10(j[1])
            c1+=(1+log10(j[1]))**2
        cosine_doc[i]=c1
    return tfidf_doc,cosine_doc

#apply ltc scheme to all the queries
def tfidf_query_vector(query,vec_df,N):
    tfidf_query=[]
    cosine_query=[]
    for i in range(len(query)):
        temp_list={}
        c1=0
        for j in query[i][1]:
            qtf =( (1+log10(j[1])) * log10(N/vec_df[j[0]]))
            temp_list[j[0]]=qtf
            c1+=qtf**2
        tfidf_query.append(temp_list)    
        cosine_query.append(c1)
    return tfidf_query,cosine_query


def main():
    index_file_name=sys.argv[1]
    query_file_name=sys.argv[2]
    index_file=open(index_file_name,"rb")
    read_index(index_file)
    print("index reading done")
    query_file = open(query_file_name, "r")
    ranked_docs = parseRankedListDoc() #ranked document list from part 1 2a.
    golden_rank = parseRelevantDocsList() #golden standard ranking
   
    alpha = [1, 0.5, 1]
    beta = [1, 0.5, 0.5]
    gamma = [0.5, 0.5, 0]
    corpus = getcorpus()
    query_rf1 = []
    query_rf2 = []
    query_rf3 = []
    query_psrf1 = []
    query_psrf2 = []
    query_psrf3 = []
    df=doc_freq()
    total_docs=len(corpus)
    rank_rf1={}
    rank_rf2={}
    rank_rf3={}
    rank_psrf1={}
    rank_psrf2={}
    rank_psrf3={}
    queries=[]
    for line in query_file:
        queries.append(line)

    '''
    For RF(relevance feedback only)
    For each query consider top 20 document from PAT2_ranked_list_A.csv.
    For each combination of alpha,beta and gamma store the updated query.
    '''
    for l in queries:
        q = l.split(',')
        temp_query1 = query_tokens(q[1][:-1])
        
        temp_query2 = query_tokens(q[1][:-1])
        temp_query3 = query_tokens(q[1][:-1])
        q[0]=int(q[0])
        docs = ranked_docs[q[0]]

        total_relevant = 0
        total_nonrelevant=0
       
        #calculate relevant and non relevant documents
        relevant_docs=[]
        non_relevant_docs=[]
        for j in range(0, 20):
            if q[0] in golden_rank and docs[j] in golden_rank[q[0]] and golden_rank[q[0]][docs[j]] == 2:
                total_relevant+=1
                if len(relevant_docs)==0:
                    relevant_docs=corpus[docs[j]]
                else:
                    relevant_docs=merging_lists(relevant_docs,corpus[docs[j]])
            else:
                total_nonrelevant+=1
                if len(non_relevant_docs)==0:
                    non_relevant_docs=corpus[docs[j]]
                else:
                    non_relevant_docs=merging_lists(non_relevant_docs,corpus[docs[j]])
        

        #multiply alpha to original query vector
        for k in range(len(temp_query1)):
            temp_query1[k]=(temp_query1[k][0],alpha[0]*temp_query1[k][1])
            temp_query2[k]=(temp_query2[k][0],alpha[1]*temp_query2[k][1])
            temp_query3[k]=(temp_query3[k][0],alpha[2]*temp_query3[k][1])
        
        if total_relevant>0:
            for e in range(len(relevant_docs)):
                relevant_docs[e]=(relevant_docs[e][0],relevant_docs[e][1]/total_relevant)
        if total_nonrelevant>0:
            for e in range(len(non_relevant_docs)):
                non_relevant_docs[e]=(non_relevant_docs[e][0],non_relevant_docs[e][1]/total_nonrelevant)
        
        #update the query vector by adding (beta*relevant_docs) and adding ((-1)*gamma*nonrelevant_docs)
        temp_query1=update_query(temp_query1,relevant_docs,beta[0],non_relevant_docs,gamma[0])
        temp_query2=update_query(temp_query2,relevant_docs,beta[1],non_relevant_docs,gamma[1])
        temp_query3=update_query(temp_query3,relevant_docs,beta[2],non_relevant_docs,gamma[2])
        
        
        #remove negative terms from a query vector
        updated_query1=remove_negative(temp_query1)
        updated_query2=remove_negative(temp_query2)
        updated_query3=remove_negative(temp_query3)

        query_rf1.append((q[0],updated_query1)) 
        query_rf2.append((q[0],updated_query2))
        query_rf3.append((q[0],updated_query3))
    
    '''
    For PS-RF( psuedo relevance feedback only)
    For each query consider only top 10 document from PAT2_ranked_list_A.csv.
    For each combination of alpha,beta and gamma store the updated query.
    '''
    for l in queries:
        q = l.split(',')

        temp_query1 = query_tokens(q[1][:-1])
        temp_query2 = query_tokens(q[1][:-1])
        temp_query3 = query_tokens(q[1][:-1])
        q[0]=int(q[0])
        docs = ranked_docs[q[0]]
      
        #multiply query vector by alpha
        for k in range(len(temp_query1)):
            temp_query1[k]=(temp_query1[k][0],alpha[0]*temp_query1[k][1])
            temp_query2[k]=(temp_query2[k][0],alpha[1]*temp_query2[k][1])
            temp_query3[k]=(temp_query3[k][0],alpha[2]*temp_query3[k][1])
        
        
        relevant_docs=corpus[docs[0]]
        for j in range(1, 10):
            relevant_docs=merging_lists(relevant_docs,corpus[docs[j]])

        for e in range(len(relevant_docs)):
            relevant_docs[e]=(relevant_docs[e][0],relevant_docs[e][1]/10)
        
        #only add beta*relevant_docs to query vector
        temp_query1 = update_query(temp_query1,relevant_docs, beta[0],[], 0)
        temp_query2 = update_query(temp_query2, relevant_docs, beta[1], [], 0)
        temp_query3 = update_query(temp_query3,relevant_docs, beta[2], [],0)
        
        #remove negetaive terms from query vector
        updated_query1=remove_negative(temp_query1)
        updated_query2=remove_negative(temp_query2)
        updated_query3=remove_negative(temp_query3)
       
        query_psrf1.append((q[0],updated_query1))
        query_psrf2.append((q[0],updated_query2))
        query_psrf3.append((q[0],updated_query3))

    print('done updating queries')
    
    tfidf_doc,cosine_doc=tfidf_doc_vector(corpus)  #calcuate tf-idf for all the document vector 
    
    '''
    pre-calcuate tf-idf for all queries for all combination of alpha,beta and gamma for RF and PS-RF both.
    '''
    tfidf_rf1,cosine_rf1=tfidf_query_vector(query_rf1,df,total_docs) 
    tfidf_rf2,cosine_rf2=tfidf_query_vector(query_rf2,df,total_docs)
    tfidf_rf3,cosine_rf3=tfidf_query_vector(query_rf3,df,total_docs)
    tfidf_psrf1,cosine_psrf1=tfidf_query_vector(query_psrf1,df,total_docs)
    tfidf_psrf2,cosine_psrf2=tfidf_query_vector(query_psrf2,df,total_docs)
    tfidf_psrf3,cosine_psrf3=tfidf_query_vector(query_psrf3,df,total_docs)
   
    '''
    for each query calcuate lnc_ltc score with all document and store top 20 documents for each query
    '''
    for k in range(len(queries)):
        l1=[]
        l2=[]
        l3=[]
        l4=[]
        l5=[]
        l6=[]
       
        for i in corpus:
            s1=lnc_ltc(tfidf_doc[i],cosine_doc[i],tfidf_rf1[k],cosine_rf1[k])
            s2=lnc_ltc(tfidf_doc[i],cosine_doc[i],tfidf_rf2[k],cosine_rf2[k])
            s3=lnc_ltc(tfidf_doc[i],cosine_doc[i],tfidf_rf3[k],cosine_rf3[k])
            s4=lnc_ltc(tfidf_doc[i],cosine_doc[i],tfidf_psrf1[k],cosine_psrf1[k])
            s5=lnc_ltc(tfidf_doc[i],cosine_doc[i],tfidf_psrf2[k],cosine_psrf2[k])
            s6=lnc_ltc(tfidf_doc[i],cosine_doc[i],tfidf_psrf3[k],cosine_psrf3[k])
            
            l1.append((s1,i))
            l2.append((s2,i))
            l3.append((s3,i))
            l4.append((s4,i))
            l5.append((s5,i))
            l6.append((s6,i))  

        l1.sort(reverse=True)
        l2.sort(reverse=True)
        l3.sort(reverse=True)
        
        l4.sort(reverse=True)
        l5.sort(reverse=True)
        l6.sort(reverse=True)

        rank_rf1[query_rf1[k][0]]=[]
        rank_rf2[query_rf2[k][0]]=[]
        rank_rf3[query_rf3[k][0]]=[]
        rank_psrf1[query_psrf1[k][0]]=[]
        rank_psrf2[query_psrf2[k][0]]=[]
        rank_psrf3[query_psrf3[k][0]]=[]
        

        for j in range(0,20): 

            rank_rf1[query_rf1[k][0]].append(l1[j][1]) 
            rank_rf2[query_rf2[k][0]].append(l2[j][1]) 
            rank_rf3[query_rf3[k][0]].append(l3[j][1]) 
            rank_psrf1[query_psrf1[k][0]].append(l4[j][1])
            rank_psrf2[query_psrf2[k][0]].append(l5[j][1]) 
            rank_psrf3[query_psrf3[k][0]].append(l6[j][1]) 
        print('query ',query_rf1[k][0],' processed')
    

    metric_file1 = open("PB_1_rocchio_RF_metrics.csv", "w")
    metric_file2 = open("PB_1_rocchio_PsRF_metrics.csv", "w")
    metric_file1.write('alpha,beta,gamma,mAP@20,NDCG@20\n')
    metric_file2.write('alpha,beta,gamma,mAP@20,NDCG@20\n')

    map20_rf=[0]*3
    ndcg20_rf=[0]*3
    map20_psrf=[0]*3
    ndcg20_psrf=[0]*3

    '''
    get mAP@20 and avg ndcg@20 for all combinations
    '''
    map20_rf[0],ndcg20_rf[0]=evaluate(rank_rf1,golden_rank)
    map20_rf[1],ndcg20_rf[1]=evaluate(rank_rf2,golden_rank)
    map20_rf[2],ndcg20_rf[2]=evaluate(rank_rf3,golden_rank)
    map20_psrf[0],ndcg20_psrf[0]=evaluate(rank_psrf1,golden_rank)
    map20_psrf[1],ndcg20_psrf[1]=evaluate(rank_psrf2,golden_rank)
    map20_psrf[2],ndcg20_psrf[2]=evaluate(rank_psrf3,golden_rank)
   
    for i in range(3):
        metric_file1.write(str(alpha[i])+','+str(beta[i])+','+str(gamma[i])+','+str(map20_rf[i])+','+str(ndcg20_rf[i])+'\n')
        metric_file2.write(str(alpha[i])+','+str(beta[i])+','+str(gamma[i])+','+str(map20_psrf[i])+','+str(ndcg20_psrf[i])+'\n')

    print("Document scoring and evaluation has been completed")


if __name__ == "__main__":
    main()