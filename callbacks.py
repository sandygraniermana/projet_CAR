#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dash.dependencies import Input, Output
from app import app
import dash
from layouts import D, SK, menu, choice, SC


@app.callback(
    [Output('b1', 'children'),
    Output('b2', 'children'),
    Output('b3', 'children'),
    Output('b4', 'children')],
    [Input('m', 'value')])

def display_value(value):
    return choice(value)


@app.callback(
    Output('des', 'children'),
    [Input('b1', 'value')])

def display_value(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    Output('graph_sk', 'children'),
    [Input('b2', 'value')])

def display_value(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    Output('graph_sc', 'children'),
    [Input('b3', 'value')])

def display_value(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    Output('graph_sv', 'children'),
    [Input('b4', 'value')])

def display_value(value):
    return 'You have selected "{}"'.format(value)
