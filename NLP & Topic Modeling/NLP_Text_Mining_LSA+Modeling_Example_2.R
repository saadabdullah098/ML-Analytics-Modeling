###########################
#Project Overview: Preprocess ad text from a CSV containing a corpus of documents, apply LSA topic modeling,
#                  and use Logistic Regression Modeling to Classify Documents as Relevant or Irrelevant

#Project Details: We perform basic text mining analysis utilizing bag of words and LSA topic modeling to classify ads as relevant or irrelevant. 
#                 The dataset contains 1000 ads from a website. The text content contains information extracted from the ad creatives and the ad landing pages
#                 The website owners have labeled each ad as either “relevant” (which are legitimate ads providing relevant information to the community) or 
#                 “irrelevant” (which are fraudulent, spam, or simply not relevant to the community)
#                 Based on these data, we develop a text-mining classifier that can automatically classify new ad posts as relevant or irrelevant.
#                 Ads.csv contains ads in the text column and the associated IDs in the doc_id column. It also contains labels where 1=relevant and 0=irrelevant.

#Strategy#
#STEP 1: Install Packages and Load Files
#STEP 2: Text Pre-processing
#STEP 3: Create Term-Document Matrix
#STEP 4: Convert to TF-IDF Values 
#STEP 5: LSA Topic Modeling
#STEP 6: Train Model and Make Prediction
###########################

### 1. Install Packages and Load Files

#install packages
#install.packages('tm')
#install.packages('lsa')
#install.packages('SnowballC')

#load libraries
library(tm)
library(lsa)
library(SnowballC)

#load data
mydata = read.csv('data/Ads.csv', header = TRUE, sep=',')

#Set corpus of docs to all rows and first two columns
corp = Corpus(DataframeSource(mydata[,1:2]))
#Set the label column for model training later
label = mydata$label


### 2. Text Pre-processing
corp = tm_map(corp, stripWhitespace)
corp = tm_map(corp, removePunctuation)
corp = tm_map(corp, removeNumbers)
corp = tm_map(corp, removeWords, stopwords('english'))
corp = tm_map(corp, stemDocument)


### 3.Create Term-Document Matrix
tdm = TermDocumentMatrix(corp)
#check the first 30 terms 
tdm$dimnames$Terms[1:30]


### 4.Convert Frequency Values to TF-IDF Values 
tfidf = weightTfIdf(tdm)


### 5.Apply Topic Modeling: Latent Semantic Analysis (Converts to Document-Concept Matrix)

#apply LSA to get the 20 most relevant topics 
lsa.tfidf = lsa(tfidf, dim=20)
lsa.tfidf$dk[1:10,]

#first convert to regular matrix then into a dataframe
words.df = as.data.frame(as.matrix(lsa.tfidf$dk))

### 6.Train Logistic Regression Model on Data
set.seed(1111)
train.indx = sample(1:1000, 800) 
train.data = cbind(label=label[train.indx], words.df[train.indx, ])
test.data = cbind(label=label[-train.indx], words.df[-train.indx, ])

logit.res = glm(label~., family=binomial(link=logit), data=train.data)

## Analyze results to understand if some features should be removed from modeling
summary(logit.res)

## Retrain model with removed features if needed and compare 

## Predict and analyze predictions

pred = predict(logit.res, newdata=test.data, type='response')
yhat = rep(1, nrow(test.data))
yhat[pred<0.5] = 0 

confusion = table(yhat, test.data$label)
confusion
sum(diag(confusion)) / sum(confusion)

###Save LSA File for Future Use
save(lsa.tfidf, file = 'lsa_posts.Rda')

