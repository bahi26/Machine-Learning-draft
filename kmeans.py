
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import operator

def getProfessionalTages(t,id):
    indices1 = [i for i, value in enumerate(t['answers_author_id']) if value == id]
    p = t['tags_tag_name'][indices1]
    comm=np.unique(list(p))
    return comm

#elbow method to chosse the best number of clusters
def elbow(df1):
    distortions = []
    k = 50
    K = []
    while k < 101:
        print(k)
        K.append(k)
        kmeanModel = KMeans(n_clusters=k).fit(df1)
        kmeanModel.fit(df1)
        distortions.append(sum(np.min(cdist(df1, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df1.shape[0])
        k += 2

    # # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return


#read the data
answers=pd.read_csv("data-science-for-good-careervillage/answers.csv")

for index, item in enumerate(answers['answers_author_id']):
    if item == 'b':
        print('yes')


professionals = pd.read_csv("data-science-for-good-careervillage/professionals.csv")

input_tags=pd.read_csv("data-science-for-good-careervillage/tags.csv")
tag_questions= pd.read_csv("data-science-for-good-careervillage/tag_questions.csv")

stop_tags = ['college', 'career', 'college-major ', 'career-counseling', 'scholarships', 'jobs', 'college-advice',
             'double-major', 'chef', 'college-minor', 'college-applications', 'college-student', 'school',
             'college-admissions', 'career-choice', 'university', 'job', 'college-major', 'any', 'student',
             'professional', 'graduate-school', 'career-path', 'career-paths', 'college-majors', 'career-details',
             'work', 'college-bound', 'success',  'first-job', 'life', 'classes', 'resume', 'job-search']

#merge the data
t = pd.merge(input_tags, tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id')
t = pd.merge(t,answers,left_on='tag_questions_question_id',right_on='answers_question_id')
tags=t['tags_tag_name']

#get features form the question using tfidf vectirization
vectorizer = TfidfVectorizer(max_df=0.5, max_features=100, min_df=2, stop_words='english',use_idf=True)
vec = vectorizer.fit(tags)
vectorized=vec.transform(tags)
df1=vectorized.toarray()

#cluster the data using k means
km = KMeans(n_clusters=62, init='k-means++', max_iter=100, n_init=1).fit(df1)
labels = km.labels_.tolist()

#now predict the new professional
ids=['58fa5e95fe9e480a9349bbb1d7faaddb','f1cc078488fa49b2827a9671ab1cc582','d6f7ebc104b6457fb76daf0df23fdb67']
most_common=[]
for id in range(len(ids)):
    #print(id)
    out=getProfessionalTages(t,ids[id])
    out = vec.transform(out)
    out=out.toarray()
    pred=km.predict(out)
    indices = [i for i, value in enumerate(labels) if value in pred]
    #print(indices)
    p=t['answers_author_id'][indices]
    comm=Counter(p)
    n=10
    if len(comm)<10:
        n=len(comm)
    arr=[]
    comm=comm.most_common(n)
    for a,b in comm:
        arr.append(a)
    most_common.append(arr)
most_common
