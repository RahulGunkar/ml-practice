import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

def get_data(filename):
    df = pandas.read_csv(filename)
    df['FullDescription'] = df['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)
    df['LocationNormalized'].fillna('nan', inplace=True)
    df['ContractTime'].fillna('nan', inplace=True)
    return df

train_fn = 'salary-train.csv'
test_fn = 'salary-test-mini.csv'

train = get_data(train_fn)
test = get_data(test_fn)


vectorizer = TfidfVectorizer(min_df=5)
enc = DictVectorizer()
clf = Ridge(alpha=1.0, random_state=241)

# train data set
X_train = vectorizer.fit_transform(train['FullDescription'])
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_train = hstack([X_train, X_train_categ])

# train the model
clf.fit(X_train, train['SalaryNormalized'])


# test data set
X_test = vectorizer.transform(test['FullDescription'])
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test, X_test_categ])

# prediction ...    
rslt = clf.predict(X_test)
