###########################
#Project Overview: Text mining analysis with a focus on LDA Topic Modeling

#Project Description: We perform probabilistic topic modeling by applying Latent Dirichlet Allocation (LDA) to analyze a collection of speeches from the State of the Union Address from various presidents.
#                     We analyze the texts and extract main topics from the speeches. It provides us with a useful tool to automatically 
#                     manage and understand a large volume of speeches, which could help detect relevant speeches on interested topics in the future.
#                     SOTU.csv contains the speeches in the text column and the associated IDs in the doc_id column.

#Strategy#
#STEP 1: Install Packages and Load Files
#STEP 2: Text Pre-processing
#STEP 3: Create Document-Term Matrix
#STEP 4: Topic Modeling: LDA 
#STEP 5: Topic Analysis
###########################

### 1.Install and Load Files

##Uncomment these if you need to install it
#install.packages('tm') #contains basic text processing and DT-matrix
#install.packages('topicmodels') #contains LDA model
#install.packages('wordcloud') #creates world cloud of topics extracted

library (tm)
library(topicmodels)
library(wordcloud)

#read data and create a Corpus of documents (automatically detects text column as text contents)
textdata = read.csv('data/SOTU.csv')
corp = Corpus(DataframeSource(textdata))

### 2.Text Preprocessing

#remove whitespace, punctuation, numbers, and stopwords followed by stemming
processedCorp = tm_map(corp, stripWhitespace)
processedCorp = tm_map(processedCorp, removePunctuation)
processedCorp = tm_map(processedCorp, removeNumbers)
processedCorp = tm_map(processedCorp, removeWords, stopwords('english'))
processedCorp = tm_map(processedCorp, stemDocument)

### 3.Create Document-Term(Token) Matrix

#the control is specifying the lower and upper bounds of the frequency of terms to be included in the DT-Matrix
#for this project, we only include terms that have a frequency of atleast 5
#this creates a sparse matrix where only the coordinates of non-zero values are present to reduce memory usage
DTM = DocumentTermMatrix(processedCorp, control = list(bounds = list(global = c(5,Inf))))
dim(DTM) #dimension of matrix
nTerms(DTM) #number of terms in the matrix
nDocs(DTM) #number of docs in the matrix
DTM$dimnames$Terms[1:50] #first 50 terms in the matrix

#Remove empty documents from document term matrix. There will be empty docs due to removal of stop words and frequency filter when creating the DT-Matrix
row.indx = slam::row_sums(DTM) > 0 
DTM = DTM[row.indx, ]
textdata = textdata[row.indx, ]
dim(DTM) #number of documents should decrease


### 4.Topic Modeling: LDA

set.seed(1000)
#parameters: K=20, method = Gibbs (MCMC method and use Gibbs algorithm for sampling), inter=1000 (# of iterations MCMC sampling will run)
tm = LDA(DTM, 20, method = 'Gibbs', control=list(iter=1000, verbose =50))

#Get the posterior distribution of the unknown parameters 
tm.res = posterior(tm)

#Assign the means of the posterior distribution to beta and theta 
#beta is each topics probability distribution over all terms
beta = tm.res$terms
dim(beta)
beta[,1:5] #check first 5 columns
rowSums(beta) #should equal to 1 since they are probabilities

#theta is each documents probability distribution over all topics
theta = tm.res$topics
dim(theta)
theta[1:5, ] #check first 5 rows 
rowSums(theta)[1:10]

#Check all the topics for the top 10 terms under it 
terms(tm, 10)


### 5.Topic Analysis

##Check how each document is related to each topic

#retrieve the original contents and check probability distribution
as.character(corp[2]$content) #2nd doc in corpus

#we can see that for this document, topic 1, 12, and 19 are the most relevant
barplot(theta[2,]) #probability distribution of doc 2 over all topic

##Wordclouds of Topics to Visualize Important Topics

#Visualize Topic 1
top.term.prob = sort(beta[1,], decreasing=TRUE)[1:50] #sort the first 50 items in beta=1 in decreasing order
wordcloud(names(top.term.prob), top.term.prob, random.order=FALSE) #pass in names of top prob terms and their measures

#Visualize Topic 19
top.term.prob = sort(beta[19,], decreasing=TRUE)[1:50]
wordcloud(names(top.term.prob), top.term.prob, random.order=FALSE)

#Visualize Topic 12
top.term.prob = sort(beta[12,], decreasing=TRUE)[1:50]
wordcloud(names(top.term.prob), top.term.prob, random.order=FALSE)






