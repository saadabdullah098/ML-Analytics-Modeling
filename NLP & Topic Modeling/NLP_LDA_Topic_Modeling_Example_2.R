###########################
#Project Overview: Text mining analysis with a focus on LDA Topic Modeling

#Project Description: We perform probabilistic topic modeling by applying Latent Dirichlet Allocation (LDA) to analyze a collection of news articles on business/finance topics.
#                     The news articles were collected from an Asian news website during 2015-2016 and include business/finance news on various markets around the globe
#                     We analyze the texts and extract main topics from the news collection. It provides us with a useful tool to automatically 
#                     manage and understand a large volume of business news, which could help detect relevant news on interested topics in the future.
#                     News.csv contains the news in the text column and the associated IDs in the doc_id column.

#Strategy#
#STEP 1: Install Packages and Load Files
#STEP 2: Text Pre-processing
#STEP 3: Create Document-Term Matrix
#STEP 4: Topic Modeling: LDA 
#STEP 5: Topic Analysis
###########################


### 1.Install and Load Files

#install.packages('tm')
#install.packages('SnowballC')
#install.packages('topicmodels')
#install.packages('wordcloud')

#load libraries
library(tm)
library(SnowballC)
library(topicmodels)
library(wordcloud)

textdata = read.csv('data/News.csv', header=T, sep = ',')
corp = Corpus(DataframeSource(textdata))


### 2.Text Preprocessing

#remove whitespace, punctuation, numbers, and stopwords followed by stemming
processedCorp = tm_map(corp, stripWhitespace)
processedCorp = tm_map(processedCorp, removePunctuation)
processedCorp = tm_map(processedCorp, removeNumbers)
processedCorp = tm_map(processedCorp, removeWords, stopwords('english'))
processedCorp = tm_map(processedCorp, stemDocument)


### 3.Create Document-Term(Token) Matrix

DTM = DocumentTermMatrix(processedCorp, control = list(bounds = list(global = c(3, Inf)))) #frequency=3
dim(DTM) #dimension of matrix
nTerms(DTM) #number of terms in the matrix
nDocs(DTM) #number of docs in the matrix
DTM$dimnames$Terms[1:50] #first 50 terms in the matrix


### 4.Topic Modeling: LDA

set.seed(1000)
lda_model <- LDA(DTM, 
                 k = 20,  # Number of topics
                 method = "Gibbs",  # Method to use
                 control = list(iter = 1000,  # Number of iterations
                                verbose = 50)  # Status print every 50 iterations
)


# Extract posterior distributions
lda_terms <- posterior(lda_model)$terms
lda_topics <- posterior(lda_model)$topics

# Display information about the terms matrix
dim(lda_terms)
# Show first 5 columns of the terms matrix
lda_terms[, 1:5]

# Display information about the topics matrix
dim(lda_topics)
# Show first 5 rows of the topics matrix
lda_topics[1:5, ]



### 5.Topic Analysis

# Display the 10 most relevant terms for each topic
terms_per_topic <- terms(lda_model, 10)
print(terms_per_topic)


##Analyze document 1082

# Access the original corpus (not processed)
original_text <- as.character(corp[[1082]])
cat(original_text)

# Get topic distribution for document #1082
barplot(lda_topics[1082,])

#Visualize Topic 4 which is the most relevant to this doc 
top.term.prob = sort(lda_terms[4,], decreasing=TRUE)[1:50] #sort the first 50 items in beta=1 in decreasing order
wordcloud(names(top.term.prob), top.term.prob, random.order=FALSE) #pass in names of top prob terms and their measures


#Visualize Topic 3 which is the second most relevant to this doc 
top.term.prob = sort(lda_terms[3,], decreasing=TRUE)[1:50] #sort the first 50 items in beta=1 in decreasing order
wordcloud(names(top.term.prob), top.term.prob, random.order=FALSE) #pass in names of top prob terms and their measures
