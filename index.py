#!/usr/bin/env python3
# -*- coding: utf-8 -*-
server=app.server
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
import callbacks
from layouts import D, SK, menu, SC, SV


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/describe':
        return D
    elif pathname == '/apps/graph_sklearn':
        return SK
    elif pathname == '/apps/graph_scipy':
        return SC
    elif pathname == '/apps/graph_svm':
        return SV
    # elif pathname == '/apps/hist_s':
    #     return H_sp
    else:
        return menu

# def display_page(pathname):
#     return liens()


if __name__ == '__main__':
    app.run_server(debug=True)
