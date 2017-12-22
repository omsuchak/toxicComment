import numpy as np
import pandas as pd
from sklearn import *

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
sample = pd.read_csv('../input/keras-bidirectional-lstm-baseline-lb-0-051/baseline.csv')

coly = [c for c in train.columns if c not in ['id','comment_text']]
y = train[coly]
test_id = test['id'].values

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow = train.shape[0]

vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english', max_features=50000)
data = vectorizer.fit_transform(df)

model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)
model.fit(data[:nrow], y)
print(1- model.score(data[:nrow], y))
sumbission = pd.DataFrame(model.predict(data[nrow:]))
submission.columns = coly
submission['id'] = test_id

sumbission.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sample, submission, how='left', on='id')
for c in coly:
    blend[c] = blend[c] * 0.9 + blend[c+'_'] * 0.1
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
final = blend[sample.columns]
final.to_csv('submission.csv', index=False)
