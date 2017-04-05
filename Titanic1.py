
# coding: utf-8

# In[8]:

#Exploratory data analysis
# package pandas
import pandas as pd
get_ipython().magic('pylab inline')
import os #directory path change/view current directory

#current working directory
os.getcwd()
os.chdir('/home/harish/PycharmProjects/Kaggle')
os.getcwd()

#train dataset
dataframe = pd.read_csv("train.csv")
dataframe.head(5)




# In[11]:

get_ipython().magic('quickref')


# In[10]:

get_ipython().magic('timeit os.getcwd()')


# In[2]:

dataframe.describe()


# In[3]:

#fill the missing value
dataframe['Age'].fillna(dataframe['Age'].median(), inplace = True)
dataframe.describe()


# In[12]:

get_ipython().magic('magic')


# In[4]:

survived_sex = dataframe[dataframe['Survived']==1]['Sex'].value_counts()
survived_sex
dead_sex = dataframe[dataframe['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','DEad']
df.head()
df.plot(kind='bar')
df.plot(kind='barh')
df.plot(kind='bar', stacked=True, figsize=(13,8))


# In[5]:

#Age correlation
figure = plt.figure(figsize=(13,8))
plt.hist([dataframe[dataframe['Survived']==1]['Age'],dataframe[dataframe['Survived']==0]['Age']], 
         color=['b','r'], bins=30, label=['Survived','Dead'],stacked=True)
plt.xlabel('Age')
plt.ylabel('no. of Passengers')
plt.legend()


# In[13]:

#Fare ticket vs survival -not fare :(
figure = plt.figure(figsize=(13,8))
plt.hist([dataframe[dataframe['Survived']==1]['Fare'],dataframe[dataframe['Survived']==0]['Fare']],
        bins = 30,label=['Survived','Dead'],stacked = True)
plt.xlabel('Fare price')
plt.ylabel('No. of passengers')
plt.legend()


# In[7]:

#combine age,fare, survival
plt.figure(figsize=(13,8))
ax = plt.subplot()
ax.scatter(dataframe[dataframe['Survived']==1]['Age'],dataframe[dataframe['Survived']==1]['Fare'], c='g',s=40)
ax.scatter(dataframe[dataframe['Survived']==0]['Age'],dataframe[dataframe['Survived']==0]['Fare'], c='r',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('Survived','dead'))


# In[8]:

ax = plt.subplot()
ax.set_xlabel('Average fare')
dataframe.groupby('Pclass').mean()['Fare'].plot(kind = 'barh',figsize=(13,8),ax=ax)


# In[9]:

#Embarkation
survived_embark = dataframe[dataframe['Survived']==1]['Embarked'].value_counts()
dead_embark = dataframe[dataframe['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','DEad']
df
df.plot(kind='bar',stacked=True,figsize=(13,8))


# In[10]:

#Part2:Feature Engineering
def status(feature):
    print 'processing', feature,':ok'


# In[11]:

def get_combined_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    print test.shape
    
    print train.shape
    targets=train.Survived
    train.drop('Survived',1,inplace=True)
    
    #merging the test and train data for future
    combined = train.append(test)
    #data frame from which I remove some rows. As a result,a data frame in which index is 
    #something like that: [1,5,6,10,11] is reset to [0,1,2,3,4]
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    return combined

combined = get_combined_data()
combined.shape

    


# In[12]:

#Titles extract and match it with our own values
def get_titles():

    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
get_titles()
combined.tail()


# In[13]:

combined['Age'].isnull().value_counts()


# In[14]:

#how to fill in the missing value:
grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.mean()


# In[15]:

#Filling in the nan values of age with each media
def process_age():
    
    global combined
    
    # a function that fills the missing values of the Age variable
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')
process_age()


# In[16]:

combined.info()


# In[17]:

#Name colunm is dropped and instead title column is used
def process_names():
    
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)
    
    status('names')
process_names()


# In[18]:

combined.head()


# In[19]:

def process_fares():
    
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)
    
    status('fare')
process_fares()


# In[20]:

#Processing cabin
def process_cabin():
    
    global combined
    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    
    combined = pd.concat([combined,cabin_dummies],axis=1)
    
    combined.drop('Cabin',axis=1,inplace=True)
    
    status('cabin')
process_cabin()
combined.head()


# In[21]:


#processing embarked
def process_embarked():
    
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    
    status('embarked')
process_embarked()


# In[ ]:




# In[22]:

#processing sex
def process_sex():
    
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    
    status('sex')


process_sex()


# In[23]:

#Processing Pclass
def process_pclass():
    
    global combined
    # encoding into 3 categories:(note only one column value will be 1 others are 0)
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    print pclass_dummies
    
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    
    # removing "Pclass"
    
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('pclass')

process_pclass()


# In[24]:

#processing ticket
def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')
process_ticket()


# In[25]:

combined.shape
combined.head()


# In[26]:

#processing family
def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    
    status('family')
    

process_family()



# In[27]:

combined.shape


# In[28]:

combined.head()


# In[29]:

#Normalize the features
def scale_all_features():
    
    global combined
    
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print 'Features scaled successfully !'
scale_all_features()


# In[30]:

combined.head()


# In[31]:

#Modeling :)


# In[32]:

#imoorting models:

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score



# In[33]:

#5fold cv socring function
def compute_score(clf,X,y,scoring='accuracy'):
    xval = cross_val_score(clf,X,y,cv=5,scoring=scoring)
    return np.mean(xval)
    


# In[34]:

#train test  target recovery
def recover_train_test_target():
    global combined
    train0 = pd.read_csv('train.csv')
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    
    return train,test,targets


# In[35]:

train, test, targets = recover_train_test_target()


# In[36]:

#feature selection:
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)


# In[37]:

features = pd.DataFrame()
features['feature'] = train.columns
features['importance']= clf.feature_importances_


# In[38]:



features.sort(['importance'],ascending=False)



# In[39]:

model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
train_new.shape


# In[40]:

test_new = model.transform(test)
test_new.shape


# In[41]:

#hyperparameters tuning
forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': range(200,300,10),
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# In[43]:



pipeline = grid_search



# In[45]:



output = pipeline.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)


