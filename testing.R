library(tm)
library(caret)
library(ggplot2)

# function for processing documents inside file
processFile <- function(filepath) {
  con = file(filepath, "r")
  resultVector <- c()
  x <- character()
  while ( length(line) != 0 ) {
    line = readLines(con, n = 1, encoding = "latin1")
    line = gsub("&#\\d+;", "", line)
    x = paste(x, line)
    if ( length(line) == 0 || (length(line) != 0 && grepl("</REUTERS>", line))){
      resultVector <- union(resultVector, c(x))
      x <- character()
    }
  }
  
  close(con)
  return(resultVector)
}

# going through every .sgm file in give-n directory
files <- list.files(path="Dataset/reuters21578/test", pattern="*.sgm", full.names=T, recursive=FALSE)
allDocs <- c()
for (i in 1:length(files)){
  docsOfFile <- processFile(files[i])
  allDocs <- union(allDocs, docsOfFile)
}

# creating corpus
vs <- VectorSource(allDocs)
vc <- VCorpus(vs, readerControl = list(reader = readReut21578XMLasPlain))

# remove documents with empty content
vc <- tm_filter(vc, FUN = function(x){length(x$content) > 0})

# removing unnecessary meta attributes
removeMetaAttributes <- function(x){
  PlainTextDocument(x, 
                    id = meta(x, "id"), 
                    topics = meta(x, "topics"), 
                    topics_cat = meta(x, "topics_cat"),
                    lewissplit = meta(x, "lewissplit"),
                    cgisplit = meta(x, "cgisplit"))
}

vc <- tm_map(vc, removeMetaAttributes)

# observe documents with and without set topic
topic <- meta(vc, "topics")
topic <- unlist(topic, use.names=FALSE)
topicDt <- as.data.frame(table(topic))

# observe splits to train and test sets according to lewis
ls <- meta(vc, "lewissplit")
ls <- unlist(ls, use.names=FALSE)
lsDt <- as.data.frame(table(ls))

# observe splits to train and test sets according to cgi
cgi <- meta(vc, "cgisplit")
cgi <- unlist(cgi, use.names=FALSE)
cgiDt <- as.data.frame(table(cgi))

# PREPROCESSING of text contents
vcProcessed <- vc
# transform content of documents to lower case
vcProcessed <- tm_map(vcProcessed, content_transformer(tolower))
# transform content of documents by removal of numbers
vcProcessed <- tm_map(vcProcessed, removeNumbers)
# transform content of documents by removal of punctuation
vcProcessed <- tm_map(vcProcessed, removePunctuation)
# transform content of documents by removal of stopwords
vcProcessed <- tm_map(vcProcessed, removeWords, stopwords("en"))
# remove word 'reuter' at the end of content
vcProcessed <- tm_map(vcProcessed, removeWords, c("reuter"))
# transform content of documents by removal of multispaces
vcProcessed <- tm_map(vcProcessed, stripWhitespace)

# remove unused structures
remove(allDocs, cgi, docsOfFile, files, i, ls, topic, cgiDt, lsDt, topicDt, vc, vs)

# filter documents with wanted topics
wantedTopics <- list("earn", "acq", "money-fx", "crude", "grain")
wanted <- tm_filter(vcProcessed, FUN = function(x){length(meta(x)[["topics_cat"]]) == 1})
wanted <- tm_filter(wanted, FUN = function(x){any(meta(x)[["topics_cat"]] %in% wantedTopics)})
wanted <- tm_filter(wanted, FUN = function(x){meta(x)[["lewissplit"]] %in% c("TRAIN", "TEST")})

# create dataframe with text contents, topic and indication of presence in train or test set
dataframe <- data.frame(text=unlist(sapply(wanted, `[`, "content")),
                        topic=unlist(sapply(wanted, meta, "topics_cat")),
                        train_test=(sapply(wanted, meta, "lewissplit")),
                        stringsAsFactors=F)

# create Document Term Matrix for tf and tf-idf
sourceData <- VectorSource(dataframe$text)
corpus <- Corpus(sourceData)
dtm <- DocumentTermMatrix(corpus)
weightedDtm <- weightTfIdf(dtm)

# remove sparse terms
dtm <- removeSparseTerms(dtm, 0.97)
weightedDtm <- removeSparseTerms(weightedDtm, 0.97)

# Document Term Matrix to dataframe conversion
dataframeDtm <- data.frame(as.matrix(dtm))
dataframeWeightedDtm <- data.frame(as.matrix(weightedDtm))

# split to train/test data frames
dataframeDtmTrain <- dataframeDtm[which(dataframe$train_test == "TRAIN"),]
dataframeDtmTest <- dataframeDtm[which(dataframe$train_test == "TEST"),]
dataframeWeightedDtmTrain <- dataframeWeightedDtm[which(dataframe$train_test == "TRAIN"),]
dataframeWeightedDtmTest <- dataframeWeightedDtm[which(dataframe$train_test == "TEST"),]

# set topic column for data frames
dataframeDtmTrain$topic <- dataframe$topic[which(dataframe$train_test == "TRAIN")]
dataframeDtmTest$topic <- dataframe$topic[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTrain$topic <- dataframe$topic[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTest$topic <- dataframe$topic[which(dataframe$train_test == "TEST")]

# select subset of dataframes
dataframeWeightedDtmTrain <- dataframeWeightedDtmTrain[1:1000,]
dataframeWeightedDtmTest <- dataframeWeightedDtmTest[1:400,]
dataframeDtmTrain <- dataframeDtmTrain[1:1000,]
dataframeDtmTest <- dataframeDtmTest[1:400,]

#KNN
ctrl <- trainControl(method="repeatedcv",number = 10, repeats = 3)

knnTf <- train(topic ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfidf <- train(topic ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)

knnPredict <- predict(knnTf, newdata = dataframeDtmTest)
knnTfidfPredict <- predict(knnTfidf, newdata = dataframeWeightedDtmTest)

knnTf
knnTfidf


# SVM
ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3)

svmTfLinear  <- train(topic ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinear  <- train(topic ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")

svmTfRadial  <- train(topic ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmRadial")
svmTfidfRadial  <- train(topic ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmRadial")

svmTfLinearPredict <- predict(svmTfLinear, newdata = dataframeDtmTest)
svmTfidfLinearPredict <- predict(svmTfidfLinear, newdata = dataframeWeightedDtmTest)
svmTfRadialPredict <- predict(svmTfRadial, newdata = dataframeDtmTest)
svmTfidfRadialPredict <- predict(svmTfidfRadial, newdata = dataframeWeightedDtmTest)

svmTfLinear
svmTfidfLinear
svmTfRadial
svmTfidfRadial


# Decision Tree
ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3)

treeTf  <- train(topic ~ . , data = dataframeDtmTrain, method = "rpart", trControl = ctrl )
treeTfidf  <- train(topic ~ . , data = dataframeWeightedDtmTrain, method = "rpart", trControl = ctrl )

treeTfPredict <- predict(treeTf, newdata = dataframeWeightedDtmTrain)
treeTfidfPredict <- predict(treeTfidf, newdata = dataframeWeightedDtmTrain)

treeTf
treeTfidf

#SAMOSTATNY MODEL pre kazdy TOPIC, dokument ho ma alebo nema
