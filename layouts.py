#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import dash_table
import plotly.graph_objects as go
from car import sk, sc, Svm


car_data = pd.read_csv('/home/sandy/Documents/laplateforme/projet_CAR/carData.csv')


def choice(value):
    if value == 'describe_' :
        return D
    elif value == 'graph2':
        return SC
    elif value == 'graph3':
        return SV
    else:
        value == 'graph1'
        return SK


menu = html.Div([
    html.H1(children='Projet : Achat d’une voiture', style={
            'textAlign': 'center',
            'text': '#7FDBFF'
        }),

    html.Div(children='''
        Les donnees contenues dans un csv ont des informations sur une grande variété de voitures, y compris leur prix.

        But : obtenir une bonne affaire sur une nouvelle voiture. En particulier, déterminer exactement combien payer pour un type de voiture spécifique.

        La question est de savoir comment utiliser les données pour déterminer combien payer ?
    '''),
    html.Div(id ='m', children =(
    dcc.Link(html.Button('Presentation des données', id='b1', n_clicks=0, value='describe_'), href='/apps/describe'),
    dcc.Link(html.Button('graphe scipy', id='b3', n_clicks=0, value='graph2'), href='/apps/graph_scipy'),
    dcc.Link(html.Button('graphe sklearn', id='b2', n_clicks=0, value='graph1'), href='/apps/graph_sklearn'),
    dcc.Link(html.Button('graphe svm', id='b4', n_clicks=0, value='graph3'), href='/apps/graph_svm')),
),
    html.H2(children='Conclusion', style={
            'text': '#7FDBFF'
        }),

    html.Div(children='''
    La régression linéaire ajuste un modèle linéaire avec des coéfficients w pour minimiser la somme résiduelle des carrés entre les données prédites et les données réelles par l'approximation linéaireself.
    Elle minimise l'erreur totale.
    Tandis que le SVM va choisir la séparation la plus nette possible entre deux classes, plus adaptée pour les modèles complexes.
    Elle minimise l'erreur de marge.
    Dans notre cas, la régression linéaire est légèrement plus adaptée.
    ''')

])


D = html.Div([
    html.H1(children='Presentation des donnees'),

    html.Div(children='''
        *********************************************************************
    '''),
    dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in car_data.columns],
    data=car_data.to_dict('records')),

    html.Div(id = 'des'),
    dcc.Link(html.Button('Presentation des données', id='b1', n_clicks=0, value='describe_'), href='/apps/describe')
])



SK = html.Div([
    html.H1(children='Graphe de regression lineaire avec Sklearn'),

    html.Div(children='''
        Representation des prix de ventes en fonction de l'annee.
    '''),
    dcc.Graph(
    id='plot_sk',
    figure = sk()),
    html.Div(id = 'graph_sk'),
    dcc.Link(html.Button('graphe sklearn', id='b2', n_clicks=0, value='graph1'), href='/apps/graph_sklearn')
])



SC = html.Div([
    html.H1(children='Graphe de regression lineaire avec Scipy'),

    html.Div(children='''
    Representation des prix de ventes en fonction de l'annee.
    '''),
    dcc.Graph(
    id='plot_sc',
    figure = sc()),
    html.Div(id = 'graph_sc'),
    dcc.Link(html.Button('graphe scipy', id='b3', n_clicks=0, value='graph2'), href='/apps/graph_scipy')
])


SV = html.Div([
    html.H1(children='SVM'),

    html.Div(children='''
    Representation des prix de ventes en fonction de l'annee.
    '''),
    dcc.Graph(
    id='plot_sv',
    figure = sc()),
    html.Div(id = 'graph_sv'),
    dcc.Link(html.Button('graphe svm', id='b4', n_clicks=0, value='graph3'), href='/apps/graph_svm')
])


#minilise erreur totale : meilleure
#svm : minimise erreur de marge
