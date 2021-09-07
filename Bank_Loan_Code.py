import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

df1 = pd.read_csv("bank_final.csv")
print(df1.describe())
print(df1.columns)
print(df1.nunique())

print(df1.isna().sum())
df1[df1.duplicated()] #15 duplicates
df1.drop_duplicates(inplace=True) #Duplicate records removed
df1=df1.reset_index(drop=True)

#stripping $ and , sign from currency columns and converting into float64
currency_cols = ['DisbursementGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv','BalanceGross']
for cols in currency_cols:
	df1[cols] = df1[cols].str.replace('$', '').str.replace(',', '')

#OUTPUT VARIABLE ---> MIS_STATUS

print(df1.MIS_Status.value_counts())
#so thid has 2 categorie so this is the best variable for target variable
print(df1.MIS_Status.isna().sum())
df1.drop(df1[df1.MIS_Status.isna()].index,inplace=True)

#Null values imputed

#creating output variable
df1['MIS_Status']=df1['MIS_Status'].map({'P I F':0,'CHGOFF':1})

plt.rcParams.update({'figure.figsize':(12,8)})
plt.show()
sns.countplot(x='MIS_Status',data=df1)
plt.show()
print(df1['MIS_Status'].value_counts())      # approx 40000 defaulters and 110000 non defaulters

len(df1['Name'].unique())
len(df1['City'].unique())
len(df1['State'].unique())
len(df1['Zip'].unique())
len(df1['Bank'].unique())
len(df1['BankState'].unique())
len(df1['CCSC'].unique())
len(df1['ApprovalDate'].unique())
len(df1['ApprovalFY'].unique())
len(df1['Term'].unique())
len(df1['NoEmp'].unique())
len(df1['NewExist'].unique()) #3 categories
len(df1['CreateJob'].unique())
len(df1['RetainedJob'].unique())
len(df1['FranchiseCode'].unique())
len(df1['UrbanRural'].unique())#0-unidentified
len(df1['RevLineCr'].unique()) #wrong entries present
len(df1['LowDoc'].unique()) #wrong entries present
len(df1['ChgOffDate'].unique())
len(df1['DisbursementDate'].unique())
len(df1['DisbursementGross'].unique())
len(df1['BalanceGross'].unique()) #only 3 unique values
len(df1['MIS_Status'].unique()) 
len(df1['ChgOffPrinGr'].unique()) 
len(df1['GrAppv'].unique())
len(df1['SBA_Appv'].unique())

# visualization to check dependency with MIS_status and other columns

####  NewExist  ####
sns.countplot(x='NewExist',data=df1)
plt.show() 
print(df1['NewExist'].value_counts()) #0-->128 wrong entries
sns.countplot(x='MIS_Status',hue='NewExist',data=df1)#Existing business more than new business in dataset
plt.show()

print(df1[['NewExist','MIS_Status']].groupby(['NewExist']).mean().sort_values(by='MIS_Status',ascending=False))
#existing business has a little more chance to default than new business
#Imputing with mode
df1.loc[(df1.NewExist !=1) & (df1.NewExist !=2),'NewExist']=1

####  FranchiseCode  ####
print(df1.FranchiseCode.isna().sum())
sns.countplot(x='MIS_Status',hue='FranchiseCode',data=df1)# only few have franchises
plt.show()

df1[['FranchiseCode','MIS_Status']].groupby(['FranchiseCode']).mean().sort_values(by='MIS_Status',ascending=True)
#defaulting chances are less for businesses with franchises

####  UrbanRural  ####
sns.countplot(x='UrbanRural',data=df1)
df1['UrbanRural'].value_counts()
plt.show()
sns.countplot(x='MIS_Status',hue='UrbanRural',data=df1)#more cases of urban; majority of unidentified is in non-default
plt.show()
df1[['UrbanRural','MIS_Status']].groupby(['UrbanRural']).mean().sort_values(by='MIS_Status',ascending=False)
#urban business more likely to default

####  RevLineCr  ####
df1['RevLineCr'].value_counts() #0-23659 , T-4819 , (`)-2 , (,)-1
sns.countplot(x='MIS_Status',hue='RevLineCr',data=df1)#RevLine of credit not availbale for majority of the businesses
plt.show()

df1.drop(df1[df1.RevLineCr.isna()].index,inplace=True)
print(df1.RevLineCr.value_counts())

df1.drop(df1[df1['RevLineCr']=='0'].index,inplace=True)
df1.drop(df1[df1['RevLineCr']=='`'].index,inplace=True)
df1.drop(df1[df1['RevLineCr']=='1'].index,inplace=True)
df1.drop(df1[df1['RevLineCr']==','].index,inplace=True)
df1.drop(df1[df1['RevLineCr']=='T'].index,inplace=True)

####  LowDoc  ####
df1['LowDoc'].value_counts() #C-83 , 1-1
sns.countplot(x='MIS_Status',hue='LowDoc',data=df1)#majority businesses are not under LowDoc
plt.show()

df1.drop(df1[df1['LowDoc']=='C'].index,inplace=True)
df1.drop(df1[df1['LowDoc']=='1'].index,inplace=True)

####  ChgOffDate  ####
df1['ChgOffDate_Yes']=0
for i in range(len(df1)):
	try:
		if len(df1['ChgOffDate'][i]):
			df1['ChgOffDate_Yes'][i]=1
	except:
		pass

pd.crosstab(df1.MIS_Status,df1.ChgOffDate_Yes)
plt.show() 
#ChgOffDate present implies it is a defaulter,and if absent, non defaulter with very few exceptions

####  BalanceGross ####
df1['BalanceGross'].value_counts() #only 2 values, rest are 0
sns.countplot(x='MIS_Status',hue='BalanceGross',data=df1)#majority businesses are not under LowDoc
plt.show()

df1[['BalanceGross','MIS_Status']].groupby(['BalanceGross']).mean().sort_values(by='MIS_Status',ascending=False)

####  ChgOffPrinGr  ####
print(df1.ChgOffPrinGr.isna().sum())
print(df1.GrAppv.isna().sum()) # no null values

pd.crosstab(df1.MIS_Status,df1.ChgOffPrinGr==0)
plt.show()
# if not defaulter then very less chance to have chargeoff amount
# if a defaulter then there are very few cases where the amount is not chargedoff

####  State  ####
sns.countplot(x='State',data=df1)
plt.show()
df1['State'].value_counts()
df1[['State','MIS_Status']].groupby(['State']).mean().sort_values(by='MIS_Status',ascending=False)
#FL state has highest probabilty to default and VT least
#Imputing the 2 NA values
a=df1.loc[df1.State.isna()]
df1.loc[df1.State.isna(),'City']
df1.loc[df1.State.isna(),'State']#JOHNSTOWN-->NY
df1.loc[df1.Zip==8070,'State']
df1.loc[df1.Name=='SO. JERSEY DANCE/MERRYLEES','State']
#PENNSVILLE-->NJ

df1.loc[df1.City=='PENNSVILLE','State']='NJ'
df1.loc[df1.City=='JOHNSTOWN       NY','State']='NY'

#Replacing the States with their probability values(Mean Encoding)
x=df1[['State','MIS_Status']].groupby(['State']).mean().sort_values(by='MIS_Status',ascending=False)
x['State']=x.index
x=x.set_index(np.arange(0,51,1))
for i in range(len(x)):
    df1=df1.replace(to_replace =x.State[i], value =x.MIS_Status[i]) 
    print(i)

####  City  ####
df1['City'].value_counts()
df1[['City','MIS_Status']].groupby(['City']).mean().sort_values(by='MIS_Status',ascending=False)

####  BankState  ####
sns.countplot(x='BankState',data=df1)
plt.show()
df1['BankState'].value_counts() #Most banks in NC least in PR
df1[['BankState','MIS_Status']].groupby(['BankState']).mean().sort_values(by='MIS_Status',ascending=False)# VA highest, MA least
#Bank in VA state has highest probabilty to default and MA least

####  ApprovalFY  ####
sns.countplot(x='ApprovalFY',data=df1)# more approvals in 1997-1998 and 2004-2007
plt.show()
df1['ApprovalFY'].value_counts() #highest no of approvals in 2006 least in 1962,65,66 
sns.countplot(x='MIS_Status',hue='ApprovalFY',data=df1)
plt.show()
df1[['ApprovalFY','MIS_Status']].groupby(['ApprovalFY']).mean().sort_values(by='ApprovalFY',ascending=True)
# if loan is approved before 1982, high probability to default; 1997-2003 very less chance to default 

len(df1.loc[(df1['ApprovalFY']<=1980)])# only 305 approvals before 1980
len(df1.loc[(df1['ApprovalFY']>1980) & (df1['ApprovalFY']<1990)])
len(df1.loc[(df1['ApprovalFY']>1990) & (df1['ApprovalFY']<2003)])
len(df1.loc[(df1['ApprovalFY']>2003)])

df1['ApprovalFY_bin']=pd.cut(df1['ApprovalFY'],bins=[1960,1980,1990,2003,2010],labels=[1,2,3,4])
sns.countplot(x='MIS_Status',hue='ApprovalFY_bin',data=df1)
plt.show()
df1[['MIS_Status','ApprovalFY_bin']].groupby(['ApprovalFY_bin']).mean().sort_values(by='ApprovalFY_bin',ascending=True)


####  ApprovalDate  ####
print(df1.ApprovalDate.isna().sum())
print(len(df1.ApprovalDate.value_counts()))
sns.countplot(x='ApprovalDate',hue='MIS_Status',data=df1)
plt.show()
df1[['MIS_Status','ApprovalDate']].groupby(['ApprovalDate']).mean().sort_values(by='ApprovalDate',ascending=True)


####  Term  ####
sorted(df1['Term'].unique()) # min=0 , max=480
sns.distplot(df1['Term'])
plt.show()
sns.boxplot(x='MIS_Status',y='Term',data=df1)
plt.show()

len(df1.loc[(df1['Term']==0)])
len(df1.loc[(df1['Term']==0) & (df1['MIS_Status']==1)]) #189/202 = 0.935
#If term = 0, almost surely defaults
len(df1.loc[(df1['Term']<=60)])
len(df1.loc[(df1['Term']<=60) & (df1['MIS_Status']==1)])
len(df1.loc[(df1['Term']>120) & (df1['Term']<=180)])
len(df1.loc[(df1['Term']>120) & (df1['Term']<=180)& (df1['MIS_Status']==1)])
len(df1.loc[(df1['Term']>300) & (df1['Term']<=360)])
len(df1.loc[(df1['Term']>300) & (df1['Term']<=360) & (df1['MIS_Status']==1)])
len(df1.loc[(df1['Term']>360)])
len(df1.loc[(df1['Term']>360) & (df1['MIS_Status']==1)])

df1['Term_bin']=0
df1['Term_bin']=pd.cut(df1['Term'],bins=[-1,60,120,180,240,300,360,480],labels=[1,2,3,4,5,6,7])
#cutting the dataframe to 5 year terms ie 60 months each;last bin 10 years

sns.countplot(x='MIS_Status',hue='Term_bin',data=df1)#more defaulters for 0-5 year term; more non defaulters for 5-40 year term
plt.show()
df1[['MIS_Status','Term_bin']].groupby(['Term_bin']).mean().sort_values(by='Term_bin',ascending=True)
#for 0-5 and 30-40 more chance of defaulting; for 5-30 less chance of defaulting
p=pd.DataFrame(df1[['MIS_Status','Term_bin']].groupby(['Term_bin']).mean())
plt.title("Term VS probability to default")
plt.xlabel("Five year Terms")
plt.ylabel("Probability to default")
plt.bar(p.index,p.MIS_Status,color='crimson')
plt.show()

####  NoEmp  ####
sorted(df1['NoEmp'].unique())# min=0 ; max=9999
sns.distplot(df1['NoEmp'])
sns.boxplot(x='MIS_Status',y='NoEmp',data=df1)
plt.show()

len(df1.loc[df1['NoEmp']>100]) # only 829 businesses have more than 100 employees
len(df1.loc[df1['NoEmp']<=5]) # 98194 have 5 or less employees
len(df1.loc[df1['NoEmp']<=10]) #122309
len(df1.loc[(df1['NoEmp']>30) & (df1['NoEmp']<=100)])
len(df1.loc[(df1['NoEmp']>100) & (df1['NoEmp']<=10000)])

df1['Emp_bin']=0 # Slicing number of employees into groups
emp_bin=[-1,5,10,15,20,30,100,1000,10000]
emp_lab=list(range(1,9))
df1['Emp_bin']=pd.cut(df1['NoEmp'],bins=emp_bin,labels=emp_lab)

sns.countplot(x='MIS_Status',hue='Emp_bin',data=df1)#both follow same pattern
plt.show()
df1[['MIS_Status','Emp_bin']].groupby(['Emp_bin']).mean().sort_values(by='MIS_Status',ascending=False)
# as the number of employees increase chances of default decrease
p=pd.DataFrame(df1[['MIS_Status','Emp_bin']].groupby(['Emp_bin']).mean())
plt.title("Number of Employees VS probability to default")
plt.xlabel("Number of Employees in bins")
plt.ylabel("Probability to default")
plt.bar(p.index,p.MIS_Status,color='crimson')

####  CreateJob ####
sorted(df1['CreateJob'].unique()) #min=0 ; max=3000
sns.boxplot(x='MIS_Status',y='CreateJob',data=df1)
plt.show()

len(df1.loc[df1['CreateJob']>100])# only 44 businesses create more than 100 jobs
len(df1.loc[(df1['CreateJob']>10) & (df1['CreateJob']<=100)])# only 3541 business creates jobs between 10 and 100
len(df1.loc[(df1['CreateJob']>5) & (df1['CreateJob']<=10)])# 4130
len(df1.loc[df1['CreateJob']==0]) # no jobs created for 113064 businesses

df1['CreateJob_bin']=0
df1['CreateJob_bin']=pd.cut(df1['CreateJob'],bins=[-1,0,5,10,100,400,3000],labels=[0,1,2,3,4,5])
sns.countplot(x='MIS_Status',hue='CreateJob_bin',data=df1)#same pattern
plt.show()
df1[['MIS_Status','CreateJob_bin']].groupby(['CreateJob_bin']).mean().sort_values(by='CreateJob_bin',ascending=True)
#chances of default is least when jobs created is between 10 and 400; highest when >400

####  RetainedJob  ####
sorted(df1['RetainedJob'].unique()) # min=0 ; max=9500
sns.boxplot(x='MIS_Status',y='RetainedJob',data=df1)
plt.show()

len(df1.loc[df1['RetainedJob']>100])# only 194 businesses have retained more than 100 jobs
len(df1.loc[df1['RetainedJob']<10])#135938
len(df1.loc[df1['RetainedJob']==0])#65810
len(df1.loc[(df1['RetainedJob']>100) & (df1['RetainedJob']<=400)])
len(df1.loc[df1['RetainedJob']>400])
len(df1.loc[(df1['RetainedJob']>400) & (df1['MIS_Status']==1)])#no defaulters when Retainedjobs>400

df1['RetainedJob_bin']=0
df1['RetainedJob_bin']=pd.cut(df1['RetainedJob'],bins=[-1,0,5,10,100,400,9500],labels=[0,1,2,3,4,5])
sns.countplot(x='MIS_Status',hue='RetainedJob_bin',data=df1)#if no jobs retained then they generally are biased to be non defaulters
plt.show()
df1[['MIS_Status','RetainedJob_bin']].groupby(['RetainedJob_bin']).mean().sort_values(by='RetainedJob_bin',ascending=True)
#if retained jobs=0 defaulting very less;then as the jobs increases, the chances of defaulting comes down;defaulters high for 1-10 range

####  DisbursementDate  ####

# converting Date text columns to datetime object
date_cols = ['ApprovalDate','DisbursementDate', 'ChgOffDate']
for dates in date_cols:
	df1[dates] = pd.to_datetime(df1[dates])

print(df1.ChgOffDate.isna().sum())
print(df1.shape)

# 109533/149999# 74% of data in ChgoffDate is filled with nan val value so drop this column -ChgOffDate

# droping the ChgOffDate Column
dro=['ChgOffDate']
df1.drop(dro,inplace=True,axis=1)

df1.drop(df1[df1.DisbursementDate.isna()].index,inplace=True)

di=df1.DisbursementDate.value_counts().to_dict()
df1.DisbursementDate=df1.DisbursementDate.map(di)
df1.head()

sns.countplot(x='MIS_Status',hue='DisbursementDate',data=df1)#most dates are between 1984-1998 and 2003-2008
plt.show()
df1[['MIS_Status','DisbursementDate']].groupby(['DisbursementDate']).mean().sort_values(by='MIS_Status',ascending=True)

####  DisbursementGross  ####
print(df1.DisbursementGross.isna().sum())
print(df1.DisbursementGross.value_counts())
df1.DisbursementGross.isna().sum()

####  GrAppv  ####

df1.Bank.isna().sum()# we have 147 nan values in the Bank column so we need to drop these as imputation will not be good for these nominal data
df1.drop(df1[df1.Bank.isna()].index,inplace=True)
df1.Bank.unique()
len(df1.Bank.unique())

####  SBA_Appv  ####

sns.countplot(x='MIS_Status',hue='SBA_Appv',data=df1)
plt.show()
df1[['MIS_Status','SBA_Appv']].groupby(['SBA_Appv']).mean().sort_values(by='MIS_Status',ascending=False)
#chances of default negligible when SBA_Appv = DisbursementGross; highest when SBA_Appv < DisbursementGross

len(df1.loc[df1.SBA_Appv>df1.GrAppv])#Gross approved amount never less than SBA approved    
len(df1.loc[df1.SBA_Appv<df1.GrAppv])    
len(df1.loc[df1.SBA_Appv==df1.GrAppv])#8094   

#####################################################################

df1['NewExist'].value_counts()
a=df1.loc[(df1.NewExist !=1) & (df1.NewExist !=2)]
x=df1.loc[(df1.NewExist ==1),['NoEmp', 'NewExist', 'CreateJob','RetainedJob']]

df1.loc[(df1.ApprovalFY ==2006),'NewExist'].value_counts()
sns.countplot(df1.CreateJob_bin,hue=df1.NewExist)
plt.show()
df1.loc[(df1.CreateJob_bin ==0),'NewExist'].value_counts()

sns.countplot(df1.NewExist,hue=df1.NoEmp>100)
plt.show()
sns.countplot(df1.NewExist,hue=df1.CreateJob>100)
plt.show()
#no relation found
#Hence imputing with mode

df1['LowDoc'].value_counts()
a=df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N')]
df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N'),'State'].value_counts()
df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N'),'BankState'].value_counts()
df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N'),'ApprovalFY'].value_counts()

df1.loc[(df1.ApprovalFY==1998),'MIS_Status'].value_counts()
df1.loc[(df1.State=='TX'),'MIS_Status'].value_counts()
df1.loc[(df1.ApprovalFY==2006),'LowDoc'].value_counts()#when ApprovalFY=2006 ,LowDoc never Y 

a=df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N') & (df1.ApprovalFY!=2006)]
a.State.value_counts()
a.BankState.value_counts()
a.ApprovalFY.value_counts()
a.RevLineCr.value_counts()
#No clear relation for LowDoc with any other feature
#Hence imputation done with mode

df1['RevLineCr'].value_counts()
a=df1.loc[(df1.RevLineCr!='Y')&(df1.RevLineCr!='N')&(df1.RevLineCr!='0')]
df1.loc[(df1.ApprovalFY ==1998),'RevLineCr'].value_counts()
sns.countplot(df1.UrbanRural,hue=df1.RevLineCr)
plt.show()
sns.countplot(df1.Term_bin,hue=df1.RevLineCr)#if urban,more having revline credit;if rural more not having
plt.show()
sns.countplot(df1.LowDoc,hue=df1.RevLineCr)#if under LowDoc, then no revline credit
plt.show()
a[a.LowDoc=='Y']
a=df1.loc[(df1.RevLineCr!='Y')&(df1.RevLineCr!='N')&(df1.RevLineCr!='0')&(df1.RevLineCr!='T')]

#UrbanRural-wrong values
#RevLineCr-wrong values

d=['BalanceGross','ChgOffPrinGr','SBA_Appv']
df1.BalanceGross=df1.BalanceGross.astype('float')
df1.ChgOffPrinGr=df1.ChgOffPrinGr.astype('float')
df1.SBA_Appv=df1.SBA_Appv.astype('float')

##############################################################################
####  Feature Selection  ####
df1.columns

features=['City', 'State', 'Zip', 'Bank', 'BankState', 'CCSC', 'ApprovalDate',
		'Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob',
		'FranchiseCode', 'RevLineCr', 'LowDoc', 'DisbursementDate',
		'BalanceGross', 'MIS_Status', 'ChgOffPrinGr', 'SBA_Appv']

df1_clensed=df1[features]
print(df1_clensed)

corri=df1.corr(method='pearson')
print(corri.style.background_gradient(cmap='coolwarm').set_precision(2))

cat_features=[i for i in df1_clensed.columns if df1_clensed.dtypes[i]=='object']
len(cat_features),cat_features

print(df1_clensed.City.value_counts())

dr=['City','Zip']
df1_clensed.drop(dr,inplace=True,axis=1)

c=df1_clensed.ApprovalDate.value_counts().to_dict()

df1_clensed.ApprovalDate=df1_clensed.ApprovalDate.map(c)
df1_clensed.head()

c=df1_clensed.ApprovalDate.value_counts().to_dict()

df1_clensed.ApprovalDate=df1_clensed.ApprovalDate.map(c)
df1_clensed.head()

di=df1_clensed.DisbursementDate.value_counts().to_dict()

df1_clensed.DisbursementDate=df1_clensed.DisbursementDate.map(di)
df1_clensed.head()

df_frequency_map = df1_clensed.State.value_counts().to_dict()
df1_clensed.State = df1_clensed.State.map(df_frequency_map)

f=df1_clensed.Bank.value_counts().to_dict()
df1_clensed.Bank = df1_clensed.Bank.map(f)

g=df1_clensed.BankState.value_counts().to_dict()
df1_clensed.BankState=df1_clensed.BankState.map(g)
df1_clensed.head()

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()

df1['RevLineCr']= label_encoder.fit_transform(df1['RevLineCr'])
df1_clensed['LowDoc']= label_encoder.fit_transform(df1_clensed['LowDoc'])

dr1=['RevLineCr']
df1_clensed.drop(dr1,inplace=True,axis=1)

dr2=['BankState']
df1_clensed.drop(dr2,inplace=True,axis=1)

#creating independent and dependent features
columns=df1_clensed.columns.tolist()
columns=[c for c in columns if c not in ['MIS_Status']]
target='MIS_Status'
target

state=np.random.RandomState(42)
x=df1_clensed[columns]
y=df1_clensed[target]
x.shape

corr_mat=df1_clensed.corr()
top_features=corr_mat.index
sns.heatmap(df1_clensed[top_features].corr(),annot=True,cmap='RdYlGn')
plt.show()

#Train-Test-Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
# The classes are heavily skewed we need to solve this issue.
print('No Frauds', round(df1_clensed['MIS_Status'].value_counts()[0]/len(df1_clensed) * 100,2), '% of the dataset')
print('Frauds', round(df1_clensed['MIS_Status'].value_counts()[1]/len(df1_clensed) * 100,2), '% of the dataset')

print('Distribution of the Classes in the subsample dataset')
print(df1_clensed['MIS_Status'].value_counts()/len(df1_clensed))

sns.countplot('MIS_Status', data=df1_clensed)
plt.title('Equally Distributed Classes', fontsize=10)
plt.show()

# Distribution of the Classes in the subsample dataset
# 1    0.726755
# 0    0.273245
# Name: MIS_Status, dtype: float64
print(df1_clensed.info())

from imblearn.over_sampling import SMOTE
# from imblearn import under_sampling, over_sampling
smote=SMOTE(random_state=10)

x_train_smote,y_train_smote=smote.fit_sample(x_train,y_train)

from collections import Counter
print("before SMOTE:",Counter(y_train))
print('after SMOTE:',Counter(y_train_smote))
# before SMOTE: Counter({1: 69892, 0: 26317})
# after SMOTE: Counter({1: 69892, 0: 69892})

print('Distribution of the Classes in the subsample dataset')
sns.countplot(y_train_smote, data=df1_clensed)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

cv_method = KFold(n_splits=2, shuffle=True)

############################## Model building ########################################

#XGB
xgb2 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
param = [{'max_depth':[1,2,3],'n_estimators':[5,10,25,50],'learning_rate':np.linspace(1e-16,1,3)}]
grid = GridSearchCV(estimator = xgb2,
								param_grid = param,
								scoring = 'neg_mean_squared_error',
								cv = cv_method,
								n_jobs = -1)

grid.fit(x_train_smote,y_train_smote)

y_pred=grid.predict(x_test)
from sklearn.metrics import accuracy_score 
print("ACCURACY",accuracy_score(y_test,y_pred))
pd.crosstab(y_test,y_pred)
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = grid.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('oversampling model: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='roc')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#UNDERSAMPLING

from imblearn.under_sampling import NearMiss

nm= NearMiss()
x_res,y_res=nm.fit_sample(x_train,y_train)

from collections import Counter
print("before SMOTE:",Counter(y_train))
print('after SMOTE:',Counter(y_res))

# before SMOTE: Counter({1: 69892, 0: 26317})
# after SMOTE: Counter({0: 26317, 1: 26317})

print('Distribution of the Classes in the subsample dataset')
sns.countplot(y_res, data=df1_clensed)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()
# Distribution of the Classes in the subsample dataset

cv_method = KFold(n_splits=10, shuffle=True)

xgb1 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
						colsample_bytree=1, max_depth=7)
param = [{'max_depth':[1,2,3],'n_estimators':[5,10,25,50],'learning_rate':np.linspace(1e-16,1,3)}]
grid1 = GridSearchCV(estimator = xgb1,
								param_grid = param,
								scoring = 'neg_mean_squared_error',
								cv = cv_method,
								n_jobs = -1)
grid1.fit(x_res,y_res)

y_pred1=grid1.predict(x_test)

from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test,y_pred1))
pd.crosstab(y_test,y_pred1)
plt.show()

####  Saving model in system  ####
import pickle
pickle.dump(grid1,open('model.pkl','wb'))

####  Load model  ####
grid1=pickle.load(open('model.pkl','rb'))

#########################################################################
