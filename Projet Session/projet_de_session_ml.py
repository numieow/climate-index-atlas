import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import sklearn as sk
#import seaborn as sn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


owid_data = pd.read_csv("./owid_data.csv")
gdp_data = pd.read_csv("./Croissance_PIB.csv")
df_owid = pd.DataFrame(owid_data)
df_gdp = pd.DataFrame(gdp_data)

#Récupérer les années en commun ainsi que la liste des pays en commun pour les 2 datasets
df_owid.drop(df_owid[df_owid.year < 1980].index, inplace=True) #=> Prévisions du GDP ont des données depuis 1980
pays_owid = pd.unique(df_owid['country']).tolist()
df_gdp.rename(columns={'Real GDP growth (Annual percent change)' : 'country'}, inplace=True)
pays_prev = pd.unique(df_gdp['country']).tolist()
pays_commun = [pays for pays in pays_owid if pays in pays_prev]

#On enlève les données des pays qui ne sont pas en commun dans les 2 datasets
df_owid = df_owid[df_owid['country'].isin(pays_commun)]
df_gdp = df_gdp[df_gdp['country'].isin(pays_commun)]

print("Longueur de la liste des pays communs : ", len(pays_commun))
print("Longueur de la liste des pays owid : ", len(df_gdp['country']))
print("Longueur de la liste des payx des prévisions : ", len(df_gdp['country']))

#Décrire les datasets :
#print("=== OUR WORLD IN DATA ===")
#print(df_owid.info(show_counts=1))
#print("=== PRÉVISIONS GDP ===")
#print(df_gdp.info(show_counts=1))

#On ne garde que les colonnes utiles des données Our World in Data
df_owid = df_owid[['country', 'year', 'gdp', 'co2', 'temperature_change_from_ch4', 'temperature_change_from_co2', 'temperature_change_from_ghg', 'temperature_change_from_n2o']]

#Remplissage des colonnes et création de la colonne de cumul
df_owid.bfill(inplace=True)
df_owid.ffill(inplace=True)
df_owid['total_temp_change'] = df_owid['temperature_change_from_ch4'] + df_owid['temperature_change_from_co2'] + df_owid['temperature_change_from_ghg'] + df_owid['temperature_change_from_n2o']
df_owid = df_owid[['country', 'year', 'gdp', 'co2', 'total_temp_change']]

pays_manquants = list(df_gdp[df_gdp['2023'] == 'no data']['country'])
print(pays_manquants)

#Il nous manque des prédictions pour quelques pays, on supprime les données
df_owid = df_owid[~df_owid['country'].isin(pays_manquants)]
df_gdp = df_gdp[~df_gdp['country'].isin(pays_manquants)]

#Taux de corrélation pour montrer le lien gdp cumulé et augmentation de température cumulée
print(df_owid['gdp'].corr(df_owid['total_temp_change']))

#Graphique pour visualiser la relation CO2-PIB
df_owid.plot(kind='scatter', x='gdp', y='co2', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# Valeurs prédites VS valeurs réelles
def plot_predicted_VS_true(test, predict, model_name):
  plt.plot(test[:100], predict[:100], 'ro')
  x = np.linspace(min(predict[:100]), max(predict[:100]), 100)
  y = x
  plt.plot(x, y, color="blue")
  plt.legend(["Data", "Perfection"])
  plt.ylabel('Predicted values')
  plt.xlabel('True values')
  plt.title('Predicted values VS true values (for : ' + model_name + ')')
  plt.show()

# Résidus VS valeurs prédites
def plot_residuals_VS_predicted(test, predict, model_name):
  residuals = test - predict
  plt.plot(predict[:100], residuals[:100], 'ro')
  plt.plot(predict[:100], np.zeros_like(predict[:100]), color="blue")
  plt.legend(["Data", "Perfection"])
  plt.ylabel('Residuals')
  plt.xlabel('Predicted values')
  plt.title('Residuals VS Predicted values (for : ' + model_name + ')')
  plt.show()


#On en garde que les données entre 1981 et 2017 pour s'assurer d'avoir des données complètes dans la plupart des pays
df_owid = df_owid[df_owid['year'] != 1980]
df_owid = df_owid[df_owid['year'] <= 2017]


#Graphique pour preuve de concept
s = StandardScaler()
pays_check = "World"
Y1 = s.fit_transform(np.array(df_owid[(df_owid['country'] == pays_check) & (df_owid['year'] <= 2017)]['total_temp_change']).reshape(-1, 1))
Y1 = np.array([e[0] for e in Y1])

Y2 = s.fit_transform(np.array(df_owid[(df_owid['country'] == pays_check) & (df_owid['year'] <= 2017)]['gdp']).reshape(-1, 1))
Y2 = np.array([e[0] for e in Y2])

X = df_owid[(df_owid['country'] == pays_check)  & (df_owid['year'] <= 2017)]['year']
plt.plot(X, Y1, "-b", label="total_temp_change")
plt.plot(X, Y2, "-r", label="gdp")
plt.legend(loc="upper left")
plt.show()

ser1 = pd.Series(Y1)
ser2 = pd.Series(Y2)
print(ser1.corr(ser2))

#Remplissage des données pour les années après 2018
#Création de la liste des colonnes de df_gdp (entiers en string)
colonnes_a_traverser_pour_gdp = [str(annee) for annee in [i for i in range(2018, 2029)]]
colonnes_en_entier = [int(annee) for annee in colonnes_a_traverser_pour_gdp]

#On enlève les pays pour lesquels il manque des données
for annee in colonnes_a_traverser_pour_gdp :
  df_gdp.drop(df_gdp[df_gdp[annee] == 'no data'].index, inplace=True)

#On calcule une liste finale de pays présents dans les 2 pays puis supprimons les pays superflus
liste_pays_finale_owid = pd.unique(df_owid['country']).tolist()
liste_pays_finale_gdp = pd.unique(df_gdp['country']).tolist()
liste_pays_finale = [pays for pays in liste_pays_finale_owid if pays in liste_pays_finale_gdp]
df_owid = df_owid[df_owid['country'].isin(liste_pays_finale)]
df_gdp = df_gdp[df_gdp['country'].isin(liste_pays_finale)]

#On crée les données de GDP pour les années 2018-2028
for pays in liste_pays_finale :
  for annee in colonnes_en_entier :
    valeur_n_moins_un = df_owid[(df_owid['country'] == pays) & (df_owid['year'] == annee - 1)]['gdp'].values[0]
    valeur_n_moins_un = float(valeur_n_moins_un)
    facteur_modif = float(df_gdp[df_gdp['country'] == pays][str(annee)].values[0].replace(',', '.'))
    nouveau_nombre = valeur_n_moins_un * (1 + facteur_modif/100)
    df_a_concat = pd.DataFrame([[pays, annee, nouveau_nombre, 0, 0]], columns=['country', 'year', 'gdp', 'co2', 'total_temp_change'])
    df_owid = pd.concat([df_owid, df_a_concat])
    df_owid.index = df_owid.index + 1

#On crée des dataframes pour les années avant et après 2017 pour faciliter les calculs
df_2017 = df_owid[df_owid['year'] <= 2017]
df_2018 = df_owid[df_owid['year'] > 2017]

#Le dictionnaire qui va contenir tous les modèles
d = {}

#Construction du dictionnaire
for pays in pd.unique(df_owid['country'].tolist()):
  pays_test = pays
  # Regression sur temperature globale VS gdp pays
  features = df_2017[df_2017['country'] == pays_test]['gdp'].copy() # drop labels for training set
  features = np.array(features).reshape(-1, 1)
  # print("Features :")
  # print(features)
  labels = df_2017[df_2017['country'] == pays_test]['total_temp_change'].copy()
  # print("Labels :")
  # print(labels)
  # print(len(labels))
  X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
  reg = LinearRegression().fit(X_train, Y_train)
  tab = cross_val_score(reg, features, labels, scoring="neg_mean_squared_error", cv=10)
  tab = np.sqrt(tab*-1)
  predict = reg.predict(X_test)
  predict_train = reg.predict(X_train)
  d[pays] = reg, tab, Y_test, predict, Y_train, predict_train

pays_test = "World"
# Test sur un pays
reg, tab, Y_test, predict, Y_train, predict_train = d.get(pays_test)
print("Valeurs RMSE du tableau :", tab)
print("Moyenne :", np.mean(tab))
print("Ecart-type :", np.std(tab))

# Prédiction sur les données de test
print("Modification de température prédite :")
print(predict)
print("Modification de température réelle :")
print(Y_test)
#COEFF ET VISU
print("Coefficients: \n", reg.coef_)

print("=== SUR ENSEMBLE DE TESTS ===")
# The mean squared error
print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(Y_test, predict)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f \n" % r2_score(Y_test, predict))

print("=== SUR ENSEMBLE DE TRAINING ===")
# The mean squared error
print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(Y_train, predict_train)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f \n" % r2_score(Y_train, predict_train))

#On prédit les variations de température et mettons les données dans le dataframe pour faciliter la visualisation
l = df_2018[df_2018['country'] == pays_test]["gdp"]
index_a_check = df_2018[df_2018['country'] == pays_test].index.tolist()
print(df_2018[df_2018['country'] == pays_test])
print(index_a_check)
a = [reg.predict([[gdp]])[0] for gdp in l.values]
print(a)
for i in range(len(index_a_check)) :
  df_2018.at[index_a_check[i], 'total_temp_change'] = a[i]

#Liste complète des PIB du pays testé
gdp_tot = np.concatenate((np.array(df_2017[df_2017['country'] == pays_test]['gdp']), np.array(df_2018[df_2018['country'] == pays_test]['gdp'])), axis=None)
print(gdp_tot)

#Liste complète des variations de température du pays testé, avec celles prédites
change_temp_globale_tot = np.concatenate((np.array(df_2017[df_2017['country'] == pays_test]['total_temp_change']), np.array(df_2018[df_2018['country'] == pays_test]['total_temp_change'])), axis=None)
print(change_temp_globale_tot)

s = StandardScaler()
pays_check = pays_test

#Visualisation des évolutions de PIB et de température, avec les prédictions en pointillés
Y1 = s.fit_transform(gdp_tot.reshape(-1, 1))
Y1 = np.array([e[0] for e in Y1])
Y2 = s.fit_transform(change_temp_globale_tot.reshape(-1, 1))
Y2 = np.array([e[0] for e in Y2])
X = df_owid[df_owid['country'] == pays_check]['year']

below = X < 2017
below_gdp = X < 2021
above = X >= 2016
above_gdp = X >= 2020

plt.plot(X[below_gdp], Y1[below_gdp], "-b", label="GDP réel")
plt.plot(X[below], Y2[below], "-r", label="Variation de température réelle")
plt.plot(X[above_gdp], Y1[above_gdp], "--b", label="GDP prédit")
plt.plot(X[above], Y2[above], "--r", label="Variation de température prédite")
plt.legend(loc="upper left")
plt.show()

#Visualisation de la variation de température uniquement
plt.plot(X, change_temp_globale_tot, "-r", label="Variation de température - " + pays_test)
plt.legend(loc="upper left")
plt.show()