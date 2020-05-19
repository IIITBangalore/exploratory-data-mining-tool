import pyclustering.cluster
import sklearn.cluster
import kmodes
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster import rock
from pyclustering.cluster import clarans
from pyclustering.cluster import clique
from pyclustering.cluster import ema
from pyclustering.cluster import kmedians
from pyclustering.cluster import kmedoids
from pyclustering.cluster import xmeans
from pyclustering.cluster import birch
from sklearn.cluster import AffinityPropagation
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
import numpy as np
import json, csv
from flask import Flask, render_template, url_for,flash,request,redirect
import plotly
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes


def scat2d(arr, label, hover_text, df):
    combos = list(zip(arr[:,0], arr[:,1]))
    weight_counter = Counter(combos)
    w = [weight_counter[(arr[:,0][i], arr[:,1][i])] for i, _ in enumerate(arr[:,0])]
    weights = np.sqrt(w).tolist()
    data = [go.Scatter(x=df.iloc[:,0],
                              y=df.iloc[:,1],
                              mode='markers',
                              marker=dict(color=label, size=weights, sizemode='area', sizeref=2.*max(weights)/(40.**2), showscale=True, colorscale='YlGnBu'),
                              text=df[hover_text]
                              # showlegend=True
                              )]
    layout= go.Layout(
                                    title="Title",
                                    xaxis_title=df.columns[0],
                                    yaxis_title=df.columns[1]
                              )
    fig = go.Figure(data=data, layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    clusters = graphJSON
    return clusters

def scat3d(arr, label, hover_text, df):
    combos = list(zip(arr[:,0], arr[:,1], arr[:,2]))
    weight_counter = Counter(combos)
    w = [weight_counter[(arr[:,0][i], arr[:,1][i], arr[:,2][i])] for i, _ in enumerate(arr[:,0])]
    weights = np.sqrt(w).tolist()
    data = [go.Scatter3d(x=arr[:,0],
                              y=arr[:,1],
                              z=arr[:,2],
                              mode='markers',
                              marker=dict(color=label, size=weights, sizemode='area', sizeref=2.*max(weights)/(40.**2), showscale=True, colorscale='YlGnBu'),
                              text=df[hover_text]
                              )]
    layout= go.Layout(
                                    title="Title",
                                    scene=dict(
                                    xaxis=dict(title=df.columns[0]),
                                    yaxis=dict(title=df.columns[1]),
                                    zaxis=dict(title=df.columns[2]))
                              )
    fig = go.Figure(data=data, layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    clusters = graphJSON
    return clusters

def aggl_cluster(df, n_clusters, link, hover_text):
    datadf = df.loc[:, df.columns != hover_text]
    data_list = datadf.to_numpy(dtype="int64").tolist()
    if(link == "centroid"):
        typelink = type_link.CENTROID_LINK
    elif(link == "single"):
        typelink = type_link.SINGLE_LINK
    elif(link == "complete"):
        typelink = agglomerative.type_link.COMPLETE_LINK
    else:
        typelink = agglomerative.type_link.AVERAGE_LINK
    aggl_instance = agglomerative(data_list, n_clusters, typelink)
    aggl_instance.process()
    clusters=aggl_instance.get_clusters()
    reps=aggl_instance.get_cluster_encoding()
    encoder = cluster_encoder(reps, clusters, data_list)
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    label = np.array(encoder.get_clusters(), dtype='int32')
    data_array = np.array(data_list)
    col_len = len(datadf.columns)
    if(col_len==2):
        clus = scat2d(data_array, label, hover_text, df)
        return clus
    else:
        clus = scat3d(data_array, label, hover_text, df)
        return clus

def dbscan_cluster(df, eps, neighbours, hover_text):
    datadf = df.loc[:, df.columns != hover_text]
    data_list = datadf.to_numpy(dtype="int64").tolist()
    dbscan_instance = dbscan(data_list, eps, neighbours)
    dbscan_instance.process()
    clusters=dbscan_instance.get_clusters()
    reps=dbscan_instance.get_cluster_encoding()

    encoder = cluster_encoder(reps, clusters, data_list)
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    label = np.array(encoder.get_clusters(), dtype='int32')
    data_array = np.array(data_list)
    col_len = len(datadf.columns)
    if(col_len==2):
        clus = scat2d(data_array, label, hover_text, df)
        return clus
    else:
        clus = scat3d(data_array, label, hover_text, df)
        return clus

def kmeans_cluster(df, n_clusters, tolerance, metric, hover_text):
    datadf = df.loc[:, df.columns != hover_text]
    data_list = datadf.to_numpy(dtype="int64").tolist()
    if(metric == "manhattan"):
        metric_str = distance_metric(type_metric.MANHATTAN)
    else:
        metric_str = distance_metric(type_metric.EUCLIDEAN_SQUARE)
    centers = kmeans_plusplus_initializer(data_list, n_clusters).initialize()
    kmeans_instance = kmeans(data_list, centers, tolerance, metric_str)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    reps=kmeans_instance.get_cluster_encoding()
    encoder = cluster_encoder(reps, clusters, data_list)
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    label = np.array(encoder.get_clusters(), dtype='int32')
    data_array = np.array(data_list)
    col_len = len(datadf.columns)
    if(col_len==2):
        clus = scat2d(data_array, label, hover_text, df)
        return clus
    else:
        clus = scat3d(data_array, label, hover_text, df)
        return clus

def kmodes_cluster(df, n_clusters, centroid, hover_text):
    datadf= df.loc[:, df.columns != hover_text]
    kmodes_instance = KModes(n_clusters=n_clusters, init='Huang', n_init=centroid, verbose=1)
    clusters = kmodes_instance.fit_predict(datadf)
    data_array = np.array(datadf.to_numpy().tolist())
    col_len = len(datadf.columns)
    if(col_len==2):
        clus = scat2d(data_array, clusters, hover_text, df)
        return clus
    else:
        clus = scat3d(data_array, clusters, hover_text, df)
        return clus

def kprotoypes_cluster(df, n_clusters, category, hover_text):
    datadf= df.loc[:, df.columns != hover_text]
    kmodes_instance = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2)
    clusters = kmodes_instance.fit_predict(datadf, categorical=category)
    data_array = np.array(datadf.to_numpy().tolist())
    col_len = len(datadf.columns)
    if(col_len==2):
        clus = scat2d(data_array, clusters, hover_text, df)
        return clus
    else:
        clus = scat3d(data_array, clusters, hover_text, df)
        return clus
