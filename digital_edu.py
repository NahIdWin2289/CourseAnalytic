import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def make_sex(sex):
    if sex == 1:
        return 1
    elif sex == 2:
        return 0

def make_langs(lang):
    if 'English' in lang or 'Русский'  in lang:
        return 1
    else:
        return 0
def make_occtype(typ):
    if typ == 'university':
        return 1
    elif typ == 'work':
        return 0
    else:
        return 0
def make_relation(relate):
    if relate == 0.0:
        return 0
    elif relate == 1.0:
        return 0
    elif relate == 2.0:
        return 1
    elif relate == 3.0:
        return 1
    elif relate == 4.0:
        return 1
    elif relate == 5.0:
        return 1
    elif relate == 6.0:
        return 0
    elif relate == 7.0:
        return 1
    elif relate == 8.0:
        return 1

def make_education_status(stat):
    if 'Student' in stat:
        return 1
    else:
        return 0 

def make_live_main(main):
    if main == 'False':
        return 0
    elif main == '1':
        return 0
    elif main == '0':
        return 0
    elif main == '2':
        return 1
    elif main == '3':
        return 0
    elif main == '4':
        return 1
    elif main == '5':
        return 1
    elif main == '6':
        return 1
    elif main == '7':
        return 0
    elif main == '8':
        return 1
    
df = pd.read_csv('train.csv')
df.drop(['graduation', 'bdate', 'id', 'people_main', 'city', 'occupation_name', 'last_seen'], axis = 1, inplace = True)
df['sex'] = df['sex'].apply(make_sex)
df['life_main'] = df['life_main'].apply(make_live_main)
df['relation'] = df['relation'].apply(make_relation)
df['occupation_type'] = df['occupation_type'].apply(make_occtype)
df['education_status'] = df['education_status'].apply(make_education_status)
df['langs'] = df['langs'].apply(make_langs)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop('education_form', axis = 1, inplace = True)
df.drop('career_start', axis = 1, inplace = True)
df.drop('career_end', axis = 1, inplace = True)
df.info()
print(df['langs'].value_counts())
# Женчин покупающих курс - 0.93% (влеяет)
# print(df[df['sex'] == 1]['result'].mean())
# Мужчин покупающих курс - 0.24% (влеяет)
# print(df[df['sex'] == 0]['result'].mean())

# Указали аватарку и купили - 0.54% (округлено) (не влеяет)
# print(df[df['has_photo'] == 1]['result'].mean())
# Не указали аватарку и купили - 0.54% (не влеяет)
# print(df[df['has_photo'] == 0]['result'].mean())

# Указали номер телефона и купили - 0.54% (влеяет)
# print(df[df['has_mobile'] == 1]['result'].mean())
# Не указали номер телефона и купили - 0.58(влеяет)
# print(df[df['has_mobile'] == 0]['result'].mean())

# Столбец проверен
# df.plot(x = 'followers_count', y = 'result', kind = 'scatter')
# plt.show()

# Людей в отношениях покупают - 49%(влеяет)
# print(df[df['relation'] == 1]['result'].mean())
# Людей не в отношениях покупают - 58%(влеяет)
# print(df[df['relation'] == 0]['result'].mean())

# Учащиеся люди купили - 50%
# print(df[df['education_status'] == 1]['result'].mean())
# Не учащиеся люди купили - 56%
# print(df[df['education_status'] == 0]['result'].mean())

# Людей с мировозрением в виде прогресса купило - 54%(не влеяет (почти))
# print(df[df['life_main'] == 1]['result'].mean())
# Людей с другими мировозрениями куполо - 54%(округлено)(не влеяет)
# print(df[df['life_main'] == 0]['result'].mean())

# Людей на дистант. обучении купило - 0.45%(влеяет)
# print(df[df['Distance Learning'] == 1]['result'].mean())
# Людей на полнонедельном обучении купило - 0.54%(влеяет)
# print(df[df['Full-time'] == 1]['result'].mean())
# Людей на частичном обучении купило - 0.48%(влеяет)
# print(df[df['Part-time'] == 1]['result'].mean())

x = df.drop('result', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifer = KNeighborsClassifier(n_neighbors = 3)
classifer.fit(x_train, y_train)
y_pred = classifer.predict(x_test)
percent = accuracy_score(y_test, y_pred)*100
print(y_pred, y_test, percent)