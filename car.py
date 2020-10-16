import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
import numpy as np
from numpy import polyfit
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import svm
from sklearn import metrics
import plotly.graph_objects as go


car_data = pd.read_csv('/home/sandy/Documents/laplateforme/projet_CAR/carData.csv')

c = car_data.keys()
#print(c)

d = car_data.describe()
#print(d)

#afficher le contenu des colonnes
col = car_data.values[:,1:]
#print(col)

#taille jeu de données
l = len(car_data)
#print(l)

pp = car_data['Present_Price'].values
#print(pp)
sp = car_data['Selling_Price'].values
#histogramme
# plt.hist(pp)
# plt.xlabel('present price')
# plt.ylabel('cars number')
# plt.title('Present_Price')
# plt.show()

#################################################################
#Question 3

bd = sqlite3.connect('/home/sandy/Documents/laplateforme/projet_car/base_car.db')
#
cursor = bd.cursor()
# cursor.execute('CREATE TABLE CARS (Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner)')
# bd.commit()

#car_data.to_sql('CARS', con=bd, if_exists='append', index=False)

r = cursor.execute("SELECT Present_Price FROM CARS")
res = cursor.fetchall()

# for i in res:
#     print(i)

r1 = cursor.execute("SELECT Car_Name FROM CARS")
res1 = cursor.fetchall()

# for i in res1:
#     print(i)

################################################################
#Question 4

R = cursor.execute("SELECT Present_Price, Car_Name FROM CARS")
Res = pd.DataFrame(cursor.fetchall(), columns=['Present_Price', 'Car_Name'])


#sns.catplot(x="Present_Price", y="Car_Name", data=Res)
#plt.show()

##############################################################
#regression lineaire

#numpy
#x, residues, rank, s = numpy.linalg.lstsq(a, b)

R = cursor.execute("SELECT Selling_Price, Year FROM CARS")
df = pd.DataFrame(cursor.fetchall(), columns=['Selling_Price', 'Year'])

x = df.Selling_Price
y = df.Year

parametres = np.polyfit(x, y, deg=1)
#print(parametres)
pred = np.poly1d(parametres)
p = pred(x)
# plt.scatter(x,y)
# plt.plot(x, p)
# plt.show()

######################################################
#scipy

def sc():
    y = df.Selling_Price
    x = df.Year

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    # print ("r-squared:", r_value**2)
    # plt.plot(x, y, 'o')
    # plt.plot(x, intercept + slope*x, 'r', label='fitted line')
    # plt.legend()
    # plt.show()

    s =go.Scatter(x =x, y=y, mode = 'markers')
    s1 =go.Scatter(x = x,y=intercept + slope*x, line=dict(width=2, color='rgb(255, 0, 0)'))
    l1 = go.Layout(
        xaxis={'title': 'Selling_Price'},
        yaxis={'title': 'Year'},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend={'x': 0.0, 'y': 1},
        hovermode='closest')


    return go.Figure(data=[s,s1], layout= l1)

######################################################
#sklearn

def sk():
    X = df.iloc[:, :-1].values
    Y = df.iloc[:,1].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    model = LinearRegression().fit(x_train, y_train)
    r_sq = model.score(x_train, y_train)
    #print('coefficient of determination:', r_sq)
    y_pred = model.predict(x_test)
    #
    # f = plt.scatter(x_test, y_test)
    # f = plt.plot(x_test, y_pred)
    #f = plt.show()


    d =go.Scatter(x =x_train.flatten(), y=y_train, mode = 'markers')
    d1 =go.Scatter(x = x_test.flatten(),y = y_pred,line=dict(width=2, color='rgb(255, 0, 0)'))
    l = go.Layout(
        xaxis={'title': 'Selling_Price'},
        yaxis={'title': 'Year'},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend={'x': 0.0, 'y': 1},
        hovermode='closest')


    return go.Figure(data=[d,d1], layout= l)
#print(sk())


#sklearn regression multiple

RR = cursor.execute("SELECT Selling_Price, Year, Kms_Driven, Transmission FROM CARS")
DF = pd.DataFrame(cursor.fetchall(), columns=['Selling_Price', 'Year', 'Kms_Driven', 'Transmission'])

y = DF.Year
Y = DF.iloc[:,1].values
xm = DF.Selling_Price
xm = DF.Kms_Driven
xm = DF.Transmission
XM = DF.iloc[:, :-1].values
XM = sm.add_constant(XM) # adding a constant

model = sm.OLS(Y, XM).fit()
predictions = model.predict(XM)

print_model = model.summary()
# print(print_model)


#######################################################################

#regression lineaire

S = df.Selling_Price
Ye = df.Year

def regLin(x, y):
    """
    Ajuste une droite d'équation a*x + b sur les points (x, y) par la méthode
    des moindres carrés.

    Args :
        * x (list): valeurs de x
        * y (list): valeurs de y

    Return:
        * a (float): pente de la droite
        * b (float): ordonnée à l'origine
    """
    # initialisation des sommes
    x_sum = 0.
    x2_sum = 0.
    y_sum = 0.
    xy_sum = 0.
    # calcul des sommes
    for xi, yi in zip(x, y):
        x_sum += xi
        x2_sum += xi**2
        y_sum += yi
        xy_sum += xi * yi
    # nombre de points
    npoints = len(x)
    # calcul des paramétras
    a = (npoints * xy_sum - x_sum * y_sum) / (npoints * x2_sum - x_sum**2)
    b = (x2_sum * y_sum - x_sum * xy_sum) / (npoints * x2_sum - x_sum**2)
    # renvoie des parametres
    return a, b

#print(regLin(S, Ye))


#############################################################

#SVM

def Svm():
    S = df.Selling_Price
    Ye = df.Year

    Sel = df.Selling_Price.values
    Yea = df.iloc[:,1:].values


    x_train, x_test, y_train, y_test = train_test_split(Yea, Sel, test_size=0.33, random_state=42)
    clf = svm.SVR(C=25, kernel='linear')

    clf.fit(x_train, y_train)
    P = clf.predict(x_test)
    #
    # plt.scatter(x_test, y_test)
    # plt.plot(x_test, P)
    # plt.show()
    d =go.Scatter(x =x_train.flatten(), y=y_train, mode = 'markers')
    d1 =go.Scatter(x = x_test.flatten(),y = P,line=dict(width=2, color='rgb(255, 0, 0)'))
    l = go.Layout(
        xaxis={'title': 'Selling_Price'},
        yaxis={'title': 'Year'},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend={'x': 0.0, 'y': 1},
        hovermode='closest')


    return go.Figure(data=[d,d1], layout= l)
#print(sk())



#print("Accuracy:",metrics.accuracy_score(y_test, P))
'''
# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 1)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
# b = clf.support_vectors_[-1]
# yy_up = a * xx + (b[1] - a * b[0])
'''
# plot the line, the points, and the nearest vectors to the plane
# plt.clf()
# plt.scatter(x_test, y_test, c='red')
# plt.scatter(x_test, P, c='green')
#
# # plt.plot(xx, yy, 'k-')
# # plt.plot(xx, yy_down, 'k--')
# # plt.plot(xx, yy_up, 'k--')
#
# plt.axis('tight')
# plt.show()
