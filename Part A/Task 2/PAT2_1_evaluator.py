import sys
import pandas as pd
from math import log2

# run using the command "python PAT2_1_evaluator.py ./Data/rankedRelevantDocList.xlsx PAT2_1_ranked_list_A.csv ./Data/raw_query.txt"

def parseRelevantDocsList():
    '''
    It reads parseQueryDoc.xlxs document, create a dictionary of query ids along with their relevant document list and respective relevance scores
    '''
    file_name = sys.argv[1]
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

def parseRankedListDoc():
    '''
    It reads output file of PAT_1_ranker.py, and create dictionary of query ids along with their relevant document list
    '''
    ranked_list = sys.argv[2]

    # read output file of PAT_1_ranker.py
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

def parseQueryDoc():
    '''
    It reads raw_query.txt and creates dictionary of query ids along with queries
    '''
    file_name = sys.argv[3]

    # read raw query document
    query_file = open(file_name, "r")

    # creates dictionary to store query ids along with queries
    queries = dict()

    count_num = 0
    count_title = 0

    # parsing the query file
    for line in query_file :
        line = line.lower()

        # find the start and end indices of the title and num tag, if any
        start_title = line.find('<title>') + len('<title>')
        end_title = line.find('</title>')
        start_num = line.find('<num>') + len('<num>')
        end_num = line.find('</num>')
        
        # extract the content within num tag
        if (line[0:5] == '<num>' or (start_num > 0 and end_num > 0)) and count_title == count_num:
            query_id = int(line[start_num:end_num])
            
            # storing the num id to the dictionary as key
            queries[query_id] = ''
            count_num += 1
            
        # extract the content within the title tag
        elif (line[0:7] == '<title>' or (start_title > 0 and end_title > 0)) and count_title == (count_num-1):
            query = line[start_title:end_title]

            # storing the query against the respective query id
            queries[query_id] = query.replace(',' , ' ')
            count_title += 1

    return queries

def main():
    group_no = 1

    # type of metric file 'A' or 'B' or 'C'
    K = sys.argv[2][-5]

    # output file
    metric_file_name = 'PAT2_' + str(group_no) + '_metrics_' + K + '.csv'

    metric_file = open(metric_file_name, "w")

    # storing the header in the output file
    metric_file.write('Query_ID,Query,AP@10,AP@20,NDCG@10,NDCG@20\n')

    # parsing all required documents and storing the contents
    retrieved_rankings = parseRankedListDoc()
    query_document_rankings = parseRelevantDocsList()
    queries = parseQueryDoc()

    # variables to store the mean average presision and average ndcg
    map_10 = 0
    map_20 = 0
    aver_ndcg_10 = 0
    aver_ndcg_20 = 0
    no_of_queries = len(retrieved_rankings)

    for query_id in retrieved_rankings:
        # variables to store the average presision and ndcg
        ap_10 = 0
        ndcg_10 = 0
        ap_20 = 0
        ndcg_20 = 0
        no_of_relevant_10 = 0
        no_of_relevant_20 = 0
        
        # store the relevance scores for top 10 as well as top 20 documents
        top_10_relevance = []
        top_20_relevance = []
        ideal_top_10_relevance = []
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
                
                # calculating metrics for top 10 relevant docs
                if i < 10:
                    no_of_relevant_10 += 1
                    ap_10 += no_of_relevant_10/(i+1)
                    top_10_relevance.append(relevance_score)
                    ideal_top_10_relevance.append(relevance_score)
                
                # calculating metrics for top 20 relevant docs
                no_of_relevant_20 += 1
                ap_20 += no_of_relevant_20/(i+1)
                top_20_relevance.append(relevance_score)
                ideal_top_20_relevance.append(relevance_score)
            
            # if current document id is not present in the relevant document list then relevance score is set to 0
            # non relevant documents are not considered in average precision
            else:
                # calculating metrics for top 10 relevant docs
                if i < 10:
                    top_10_relevance.append(0)
                    ideal_top_10_relevance.append(0)
                
                # calculating metrics for top 20 relevant docs
                top_20_relevance.append(0)
                ideal_top_20_relevance.append(0)
        
        # compute average precision
        if no_of_relevant_10 != 0:
            ap_10 = round(ap_10 / no_of_relevant_10, 2)
        if no_of_relevant_20 != 0:
            ap_20 = round(ap_20 / no_of_relevant_20, 2)
        
        map_10 += ap_10
        map_20 += ap_20
        
        # ideal relevance score = sorted relevance scores in descending order
        ideal_top_10_relevance.sort(reverse=True)
        ideal_top_20_relevance.sort(reverse=True)
        
        # compute the discounted cumulative gain for top 10 relevant docs
        for i in range(1, len(top_10_relevance)):
            top_10_relevance[i] = top_10_relevance[i-1] + (top_10_relevance[i] / log2(i+1))
            ideal_top_10_relevance[i] = ideal_top_10_relevance[i-1] + (ideal_top_10_relevance[i] / log2(i+1))

        # compute the discounted cumulative gain for top 20 relevant docs   
        for i in range(1, len(top_20_relevance)):
            top_20_relevance[i] = top_20_relevance[i-1] + (top_20_relevance[i] / log2(i+1))
            ideal_top_20_relevance[i] = ideal_top_20_relevance[i-1] + (ideal_top_20_relevance[i] / log2(i+1))

        # compute the normalized discounted cumulative gain 
        if ideal_top_10_relevance[9] != 0:
            ndcg_10 = round(top_10_relevance[9] / ideal_top_10_relevance[9], 2)
        if ideal_top_20_relevance[19] != 0:
            ndcg_20 = round(top_20_relevance[19] / ideal_top_20_relevance[19], 2)
        
        aver_ndcg_10 += ndcg_10
        aver_ndcg_20 += ndcg_20
        
        # write all the metrics for current query in output file
        metric_file.write(str(query_id) + ',' + queries[query_id] + ',' + str(ap_10) + ',' + str(ap_20) + ',' + str(ndcg_10) + ',' + str(ndcg_20) + '\n')

    # compute mean average precision    
    map_10 = round(map_10 / no_of_queries, 2)
    map_20 = round(map_20 / no_of_queries, 2)

    # compute average ndcg
    aver_ndcg_10 = round(aver_ndcg_10 / no_of_queries, 2)
    aver_ndcg_20 = round(aver_ndcg_20 / no_of_queries, 2)

    # write mean average precision and average ndcg for all queries in output file
    metric_file.write(',,' + str(map_10) + ',' + str(map_20) + ',' + str(aver_ndcg_10) + ',' + str(aver_ndcg_20) + '\n')
    metric_file.close()

if __name__ == "__main__":
    main()
