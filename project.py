from flask import Flask, render_template, url_for,flash,request,redirect,session
from dotenv import load_dotenv
from collections import OrderedDict
import pandas as pd
import json, csv
from apyori import apriori
from string import ascii_uppercase
import itertools
import plotly
import plotly.graph_objs as go
import math
import os
from werkzeug.utils import secure_filename
from cluster import aggl_cluster, dbscan_cluster, kmeans_cluster, kmodes_cluster, kprotoypes_cluster
from association import apriorimining, forcedir
global df_dataset,total_rows,total_cols,data_cols,cols,filename

def set_vars(filepath):
    global df_dataset,total_rows,total_cols,data_cols,cols
    df_dataset=pd.read_csv(filepath)
    total_cols=df_dataset.shape[1]
    total_rows=df_dataset.shape[0]
    cols=df_dataset.columns.values.tolist()


ALLOWED_EXTENSIONS = {'csv'}



def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_uppercase, repeat=size):
            yield "".join(s)

def set_mappings():
    idx=0
    global data_cols,cols
    data_cols={}
    for s in itertools.islice(iter_all_strings(), total_cols):
        data_cols.update({s:cols[idx]})
        idx+=1


# print("%$$$$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%")
# print(data_cols)
# print("%$$$$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%")
load_dotenv("./.env")
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/mining/<filename>/<algotype>')
def mining(filename,algotype):
    filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    session['filepath'] = filepath
    set_vars(filepath)
    set_mappings()
    if(algotype=="AR"):
        return render_template("association_rules.html",rows=total_rows, mappings=data_cols,filename=filename)
    elif(algotype=="C"):
        return render_template("cluster_algos.html",rows=total_rows, mappings=data_cols,filename=filename)


@app.route('/')
def AssignmentCurrentPage():
    return render_template("index.html",flag='0')

@app.route('/upload', methods = ['POST'])
def upldfile():
    global filename
    if request.method == 'POST':
        file = request.files['file']
    if file.filename == '':
        flash('No selected file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    set_vars(filepath)
    set_mappings()
    session['filepath'] = filepath
    return render_template("index.html",rows=total_rows,mappings=data_cols,flag='1',filename=filename)

@app.route('/choosealgo', methods=['POST'])
def choosealgo():
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    select = request.form.get('algo_select')
    return redirect(url_for("mining",filename=filename,algotype=select))

@app.route('/Apriori/<filenum>/<columns>/<supp>/<conf>/<lft>')
def AprioriRun(filenum, columns, supp, conf, lft):
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    global data_cols,df_dataset
    apriori_inputs=[supp,conf,lft]
    col_keys=columns.split(',')
    col_values=[]
    for input in col_keys:
        col_values.append(data_cols[input])

    if(filenum == '1'):
        datadf = df_dataset
        columndata = columns.split(',')
        columndata = [data_cols[c] for c in columndata]
        resultdf = datadf[columndata]
        print("$%$%$%$%$%$%$%$%$%")
        print(resultdf)
        print("$%$%$%$%$%$%$%$%$%")

    try:
        support = float(supp)
        confidence = float(conf)
        lift = float(lft)
        final_df = apriorimining(resultdf, support, confidence, lift)
    except Exception as e:
        return render_template("500.html", error=str(e))
    return render_template("table.html", rules=final_df.to_dict('records'), apriori_inputs=apriori_inputs, input_cols=zip(col_keys,col_values))

@app.route('/forcedirected/<filenum>/<columns>/<supp>/<conf>')
def FDVisualise(filenum, columns, supp, conf):
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    global data_cols,df_dataset
    if(filenum == '1'):
        datadf = df_dataset
        columndata = columns.split(',')
        # print(columndata[0])
        columndata = [data_cols[c] for c in columndata]
        resultdf = datadf[columndata]

    try:
        support = float(supp)
        confidence = float(conf)
        data_file = forcedir(columndata, resultdf, support, confidence)
    except Exception as e:
        return render_template("500.html", error=str(e))
    return render_template("force2.html", filenam=data_file)

@app.route('/PCVisualization/<filenum>/<columns>/<color>')
def PCVisualize(filenum, columns, color):
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    global data_cols,df_dataset
    cols_file = 'static/parallel.csv'
    if(filenum == '1'):
        datadf = df_dataset
        columndata = columns.split(',')
        columndata = [data_cols[c] for c in columndata]
        resultdf = datadf[columndata]
        # resultdf['color'] = datadf[data_cols[color]]

    try:
        col_len = len(columndata)
        dict_list = []
        for a in range(col_len):
            dict_list.append(dict(label=str(columndata[a]), values = resultdf[columndata[a]]))

        # data = [go.Parcats(line = dict(color = datadf[data_cols[color]], colorscale = 'rainbow', showscale = True, cmin=datadf[data_cols[color]].min(), cmax=datadf[data_cols[color]].max()), dimensions = dict_list)]
        data = [go.Parcats(line = dict(color = datadf[data_cols[color]], colorscale = 'viridis', showscale = True, cmid=datadf[data_cols[color]].median()), dimensions = dict_list)]

        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

        para = graphJSON
    except Exception as e:
        return render_template("500.html", error=str(e))
    return render_template("paraplot.html", plot=para)

@app.route('/agglomerative/<filenum>/<columns>/<nclusters>/<link>/<hover>')
def Agglomerative(filenum, columns, nclusters, link, hover):
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    global data_cols, df_dataset
    if(filenum == '1'):
        datadf = df_dataset
        columndata = columns.split(',')
        columndata = [data_cols[c] for c in columndata]
        resultdf = datadf[columndata]
        resultdf[hover] = datadf[data_cols[hover]]

    try:
        link = link.lower()
        n_clusters = int(nclusters)
        clus = aggl_cluster(resultdf, n_clusters, link, hover)
    except Exception as e:
        return render_template("500.html", error=str(e))
    return render_template("cluster.html", plot=clus,algorithm="Agglomerative")

@app.route('/DBSCAN/<filenum>/<columns>/<radius>/<neighbours>/<hover>')
def dbscanclus(filenum, columns, radius, neighbours, hover):
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    global data_cols, df_dataset
    if(filenum == '1'):
        datadf = df_dataset
        columndata = columns.split(',')
        columndata = [data_cols[c] for c in columndata]
        resultdf = datadf[columndata]
        resultdf[hover] = datadf[data_cols[hover]]

    try:
        radius = float(radius)
        neighbours = int(neighbours)
        clus = dbscan_cluster(resultdf, radius, neighbours, hover)
    except Exception as e:
        return render_template("500.html", error=str(e))
    return render_template("cluster.html", plot=clus,algorithm="DBSCAN")
    # return render_template("cluster.html", plot=clus,algorithm="K-modes")

@app.route('/Kmeans/<filenum>/<columns>/<nclusters>/<tolerance>/<metric>/<hover>')
def kcmeanclus(filenum, columns, nclusters, tolerance, metric, hover):
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    global data_cols,df_dataset
    if(filenum == '1'):
        datadf = df_dataset
        columndata = columns.split(',')
        columndata = [data_cols[c] for c in columndata]
        resultdf = datadf[columndata]
        resultdf[hover] = datadf[data_cols[hover]]

    try:
        n_clusters = int(nclusters)
        tolerance = float(tolerance)
        metric = metric.lower()
        clus = kmeans_cluster(resultdf, n_clusters, tolerance, metric, hover)
    except Exception as e:
        return render_template("500.html", errot=str(e))
    return render_template("cluster.html", plot=clus, algorithm="K-means")

@app.route('/Kmodes/<filenum>/<columns>/<nclusters>/<centroid>/<hover>')
def kmodesclus(filenum, columns, nclusters, centroid, hover):
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    global data_cols,df_dataset
    if(filenum == '1'):
        datadf = df_dataset
        columndata = columns.split(',')
        columndata = [data_cols[c] for c in columndata]
        resultdf = datadf[columndata]
        resultdf[hover] = datadf[data_cols[hover]]

    try:
        n_clusters = int(nclusters)
        centroid = int(centroid)
        clus = kmodes_cluster(resultdf, n_clusters, centroid, hover)
    except Exception as e:
        return render_template("500.html", error=str(e))
    return render_template("cluster.html", plot=clus,algorithm="K-modes")

@app.route('/Kprototypes/<filenum>/<columns>/<nclusters>/<category>/<hover>')
def kprototypeclus(filenum, columns, nclusters, category, hover):
    filepath = session.get('filepath', None)
    set_vars(filepath)
    set_mappings()
    global data_cols,df_dataset
    if(filenum == '1'):
        datadf = df_dataset
        columndata = columns.split(',')
        columndata = [data_cols[c] for c in columndata]
        resultdf = datadf[columndata]
        resultdf[hover] = datadf[data_cols[hover]]

    try:
        category_list = list(map(int, category.split(',')))
        n_clusters = int(nclusters)
        clus = kprotoypes_cluster(resultdf, n_clusters, category_list, hover)
    except Exception as e:
        return render_template("500.html", error=str(e))
    return render_template("cluster.html", plot=clus, algorithm="K-prototypes")

if __name__ == '__main__':
    # sess.init_app(app)

    app.run( host = '0.0.0.0', port = 5000 )
