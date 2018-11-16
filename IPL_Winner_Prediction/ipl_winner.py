import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns


def classification_model(model, x_train, x_test, y_train, y_test, name):
  model.fit(x_train,y_train)
  predictions = model.predict(x_test)
  accuracy = metrics.accuracy_score(predictions,y_test)
  names = [None] * len(predictions)
  j=0
  print("================", name, "===========================")
  for i in predictions:
      if i == 1:
          names[j]="Mumbai Indians"
          j=j+1
      elif i == 2:
          names[j] = "Kolkata Knight Riders"
          j=j+1
      elif i == 3:
          names[j] = "Royal Challengers Bangalore"
          j=j+1
      elif i == 4:
          names[j] = "Deccan Chargers"
          j=j+1
      elif i == 5:
          names[j] = "Chennai Super Kings"
          j=j+1
      elif i == 6:
          names[j] = "Rajasthan Royals"
          j=j+1
      elif i == 7:
          names[j] = "Delhi Daredevils"
          j=j+1
      elif i == 8:
          names[j] = "Gujarat Lions"
          j=j+1
      elif i == 9:
          names[j] = "Kings XI Punjab"
          j=j+1
      elif i == 10:
          names[j] = "Sunrisers Hyderabad"
          j=j+1
      elif i == 11:
          names[j] = "Rising Pune Supergiants"
          j=j+1
      elif i == 12:
          names[j] = "Kochi Tuskers Kerala"
          j=j+1
      elif i == 13:
          names[j] = "Pune Warriors"
          j=j+1
  newarray = x_test[['id']].values
  newarray2 = newarray.ravel()
  newdf = pd.DataFrame({'winner':predictions, 'id':newarray2})
  sns.factorplot(x="id", y="winner", data=newdf)
  plt.yticks(np.arange(0, 14, 1))
  plt.xticks(rotation='vertical')
  plt.show()
  print ('Accuracy : %s' % '{0:.3%}'.format(accuracy), name)
  

matches=pd.read_csv('/home/user/Desktop/matches.csv')
matches['winner'].fillna('Draw', inplace=True)
matches['city'].fillna('Dubai',inplace=True)
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)
encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14},
          'toss_decision': {'bat': 1, 'field': 2},
          'result': {'no result':1, 'normal':2, 'tie':3},
          'venue': {'Barabati Stadium': 1, 'Brabourne Stadium':2, 'De Beers Diamond Oval':3, 'Buffalo Park':4, 'Dr DY Patil Sports Academy':5,
                    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':6, 'Dubai International Cricket Stadium':7, 'Eden Gardens':8, 'Feroz Shah Kotla':9,
                    'Green Park':10, 'Himachal Pradesh Cricket Association Stadium':11, 'Holkar Cricket Stadium':12, 'JSCA International Stadium Complex':13,
                    'Kingsmead':14, 'M Chinnaswamy Stadium':15, 'MA Chidambaram Stadium, Chepauk':16, 'Maharashtra Cricket Association Stadium':17,
                    'Nehru Stadium':18, 'New Wanderers Stadium':19, 'Newlands':20, 'OUTsurance Oval':21, 'Punjab Cricket Association IS Bindra Stadium, Mohali':22,
                    'Punjab Cricket Association Stadium, Mohali':23, 'Rajiv Gandhi International Stadium, Uppal':24, 'Sardar Patel Stadium, Motera':25,
                    'Saurashtra Cricket Association Stadium':26, 'Sawai Mansingh Stadium':27, 'Shaheed Veer Narayan Singh International Stadium':28,
                    'Sharjah Cricket Stadium':29, 'Sheikh Zayed Stadium':30, "St George's Park": 31, 'Subrata Roy Sahara Stadium':32, 'SuperSport Park':33,
                    'Vidarbha Cricket Association Stadium, Jamtha':34, 'Wankhede Stadium':35},
          'city': {'Abu Dhabi':1, 'Ahmedabad':2, 'Bangalore':3, 'Bloemfontein':4, 'Cape Town':5, 'Centurion':6, 'Chandigarh':7, 'Chennai':8, 'Cuttack':9,
                   'Delhi':10, 'Dharamsala':11, 'Dubai':0, 'Durban':12, 'East London':13, 'Hyderabad':14, 'Indore':15, 'Jaipur':16, 'Johannesburg':17,
                   'Kanpur':18, 'Kimberley':19, 'Kochi':20, 'Kolkata':21, 'Mumbai':22, 'Nagpur':23, 'Port Elizabeth':24, 'Pune':25, 'Raipur':26, 'Rajkot':27,
                   'Ranchi':28, 'Sharjah':29, 'Visakhapatnam':30
              }
          }
matches.replace(encode, inplace=True)
matches = matches[['team1','team2','city','toss_decision','toss_winner','venue','winner', 'result', 'dl_applied', 'win_by_runs', 'win_by_wickets', 'id']]
df = pd.DataFrame(matches)
x = df[['team1','team2', 'city', 'toss_winner', 'venue', 'toss_decision', 'result', 'dl_applied', 'win_by_runs', 'win_by_wickets', 'id']]
y = df[['winner']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf1 = svm.SVC()
clf2 = tree.DecisionTreeClassifier()
clf3 = RandomForestClassifier()
clf11 = MLPClassifier()
classification_model(clf1, X_train, X_test, y_train, y_test, "SVC")
classification_model(clf2, X_train, X_test, y_train, y_test, "DecisionTreeClassifier")
classification_model(clf3, X_train, X_test, y_train, y_test, "RandomForestClassifier")
classification_model(clf11, X_train, X_test, y_train, y_test, "MLPClassifier")
