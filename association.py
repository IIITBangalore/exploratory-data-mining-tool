from apyori import apriori
import pandas as pd
import numpy as np
import math

def apriorimining(resultdf, support, confidence, lift):
    resultdf_col_list=resultdf.columns.values.tolist()
    records = []
    for i in range(0, resultdf.shape[0]):
        records.append([(resultdf_col_list[j],str(resultdf.values[i,j])) for j in range(0, resultdf.shape[1])])

    association_rules = apriori(records, min_support=support, min_confidence=confidence, min_lift=lift, min_length=2)
    results = list(association_rules)
    print(len(results))
    print("apriori computed")
    table_list =[["LHS", "RHS", "Support", "Confidence", "Lift", "FromCol", "ToCol", "Conviction", "Cosine", "Jaccard index", "Leverage"]]
    for item in results:
        pair = item[0]
        items = [x for x in pair]
        suppX = item[1]/item[2][0][2]
        suppY = item[2][0][2]/item[2][0][3]
        try:
            conviction = (1-suppY)/(1-item[2][0][2])
        except ZeroDivisionError:
            conviction = 'NaN'
        # conviction = (1-suppY)/(1-item[2][0][2])
        cosine = item[1]/math.sqrt(suppX*suppY)
        jaccard = item[1]/(suppX+suppY - item[1])
        leverage = item[1] - (suppX*suppY)
        table_list.append([str(items[0][1]), str(items[1][1]), float(item[1]), float(item[2][0][2]), float(item[2][0][3]), str(items[0][0]), str(items[1][0]), str(conviction), float(cosine), float(jaccard), float(leverage)])

    table_list.pop(0)
    final_df=pd.DataFrame(table_list, columns=["LHS", "RHS", "Support", "Confidence", "Lift", "FromCol", "ToCol", "Conviction", "Cosine", "Jaccard index", "Leverage"])
    return final_df

def forcedir(columndata, resultdf, support, confidence):
    data_file = 'static/data.json'
    l = []
    col_len = len(columndata)
    for a in range(col_len):
        l.append((resultdf[columndata[a]].unique()).tolist())

    records = []
    for i in range(0, resultdf.shape[0]):
        records.append([str(resultdf.values[i,j]) for j in range(0, resultdf.shape[1])])

    association_rules = apriori(records, min_support=support, min_confidence=confidence, min_lift=1.00000001, min_length=2)
    results = list(association_rules)
    print(len(results))
    # print("apriori computed")
    table_list =[["LHS", "RHS", "Support", "Confidence", "Lift"]]
    for item in results:
        pair = item[0]
        items = [x for x in pair]
        table_list.append([str(items[0]), str(items[1]), float(item[1]), float(item[2][0][2]), float(item[2][0][3])])

    table_list.pop(0)
    final_df=pd.DataFrame(table_list, columns=["LHS", "RHS", "Support", "Confidence", "Lift"])
    c = -1
    for t in range(len(l)):
        c = c + len(l[t])
    print(c)

    with open(data_file, 'w') as outfile:
        d = len(final_df.index) - 1
        i = 0
        outfile.write('{\n\t"nodes": [\n\t')
        for o in range(len(l)):
            for p in range(len(l[o])):
                outfile.write('{"id": "' + str(i) + '", "name": "' + str(l[o][p]) + '", "group": '+ str(o+1)+'}')
                if i == c:
                    outfile.write('\n')
                else:
                    outfile.write(',\n')
                    i = i + 1

        j = 0
        outfile.write('],\n\t"links": [\n\t')
        for index, rows in final_df.iterrows():
            rows = list(rows)
            outfile.write('{"source": "' + str(rows[0]) + '", "target": "' + str(rows[1]) + '", "value": '+ str(rows[4])+'}')
            if j == d:
                outfile.write('\n')
            else:
                outfile.write(',\n')
            j = j + 1
        outfile.write('\t]\n}')
    return data_file
