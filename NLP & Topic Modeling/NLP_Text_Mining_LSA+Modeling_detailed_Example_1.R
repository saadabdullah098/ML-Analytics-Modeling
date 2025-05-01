###########################
#Project Overview: Preprocess text from a corpus of documents, apply LSA topic modeling,
#                  and use Logistic Regression Modeling to Classify Documents as either Auto or Electronics

#Project Details: This R project implements a text classification system using Latent Semantic Analysis (LSA) to categorize documents 
#                 as either automobile or electronics related. The core of the analysis applies LSA to reduce dimensionality to 20 concepts.
#                 Followed by training a logistic regression model on the resulting document-concept matrix to predict document categories.
#                 The model's performance is evaluated using a confusion matrix.
#                 AutoElectronic.zip contains two folders containing articles referenced by IDs related to each topic: Auto and Electronics

#Strategy#
#STEP 1: Install Packages and Load Files
#STEP 2: Text Pre-processing
#STEP 3: Create Term-Document Matrix
#STEP 4: Convert to TF-IDF Values 
#STEP 5: LSA Topic Modeling
#STEP 6: Train Model and Make Prediction
###########################


### 1.Install Packages and Load Files

#install text mining package and lsa package
#install.packages('tm')
#install.packages('lsa')

#load the libraries
library(tm)
library(lsa)

#load the corpus (collection of documents), recursive=True means all subfolder files will be loaded
corp = Corpus(ZipSource('data/AutoElectronics.zip', recursive=T))

#since we have only two categories where the first 1000 observations are automobile and the next 1000 electronics
#we create a label vector to indicate this rep = repeating 1 for 100 times and 0 for 1000 times
#this will be used as the y variable 
label = c(rep(1,1000), rep(0,1000))

### 2.Text Pre-processing

corp = tm_map(corp, stripWhitespace)
corp = tm_map(corp, removePunctuation)
corp = tm_map(corp, removeNumbers)
corp = tm_map(corp, removeWords, stopwords(kind='en'))
#stemming is removing the suffixes of the same word to unify them into one common term
corp = tm_map(corp, stemDocument)

### 3.Create Term-Document Matrix

#this is stored as a sparse matrix (matrix containing lots of 0s) where only the non-zero values are stored in storage 
#i and j are the coordinates of the non-zero values and v are the actual tokens and the value represent teh # of times it appears in the document
tdm = TermDocumentMatrix(corp)
#check the first 100 terms
tdm$dimnames$Terms[1:100]

### 4.Convert Frequency Values to TF-IDF Values (Check attached notes for what a TF-IDF value is) 

#some terms appear in all documents so the idf for it becomes 0 so the total # of non-zero values decrease
tfidf = weightTfIdf(tdm)

### 5.Apply Topic Modeling: Latent Semantic Analysis (Converts to Document-Concept Matrix)

#dim = the # of concepts to extract
#contains: tk(term-concept matrix ie. how much each term is related to each of the 20 concepts), 
          #dk(document-concept matrix ie. how much each document is related to each of the 20 concepts), 
          #sk(20 largest eigenvalues retained by the analysis)
lsa.tfidf = lsa(tfidf, dim=20)

#check the dk for the first 10 docs 
lsa.tfidf$dk[1:10,]

##Create Dataframe
#organize concepts from dk as features where each document is an observation
#outcome of interest will be appended to the dataframe to serve as the y variable

#first convert to regular matrix then into a dataframe
words.df = as.data.frame(as.matrix(lsa.tfidf$dk))


### 6.Train Logistic Regression Model on Data

set.seed(1000)
train.indx = sample(1:2000, 1600) #randomly select 1600 integers to be the indicies for the training set
#attach the y variable (in column name label) using cbind
train.data = cbind(label=label[train.indx], words.df[train.indx, ])
test.data = cbind(label=label[-train.indx], words.df[-train.indx, ])

#Visualize Dataframe
fix(train.data)

#Given the binary nature of the outcomes, we can fit a binary logit model
#label is the y variables and ~. means all the rest are the x variables
logit.res = glm(label~., family=binomial(link=logit), data=train.data)


## Analyze results to understand if some features should be removed from modeling
summary(logit.res)

## Retrain model with removed features if needed and compare 

##Check the Quality of the Model
#type = response to predict the probability of success (y=1)
pred = predict(logit.res, newdata=test.data, type='response')
#if the predicted response is greater than 0.5 then we'll say it is 1 (auto) and 0 (electronic) if below 0.5
#predicted values are saved in the yhat variable

#creating a vector of 1s with lenght of test data
yhat = rep(1, nrow(test.data))
#checking pred values to see if it's below 0.5 and if it is using its index to convert that index in yhat to 0  
yhat[pred<0.5] = 0 

confusion = table(yhat, test.data$label)
confusion
#percent correctly predicted
sum(diag(confusion)) / sum(confusion)

###Save LSA File for Future Use
save(lsa.tfidf, file = 'lsa_posts.Rda')




