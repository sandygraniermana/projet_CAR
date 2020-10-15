import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
import numpy as np
from numpy import polyfit
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



car_data = pd.read_csv('/home/sandy/Documents/laplateforme/projet_car/carData.csv')

c = car_data.keys()
#print(c)

d = car_data.describe()
print(d)

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
print(parametres)
pred = np.poly1d(parametres)
p = pred(x)
# plt.scatter(x,y)
# plt.plot(x, p)
# plt.show()

######################################################
#scipy

x = df.Selling_Price
y = df.Year
#
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# print ("r-squared:", r_value**2)
# plt.scatter(x,y)
# plt.plot(x, slope)
# plt.show()

######################################################
#sklearn

X = df.iloc[:, :-1].values
Y = df.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model = LinearRegression().fit(x_train, y_train)
r_sq = model.score(x_train, y_train)
#print('coefficient of determination:', r_sq)
y_pred = model.predict(x_test)

# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_pred)
# plt.show()
