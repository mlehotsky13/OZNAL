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

# going through every .sgm file in given directory
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
vc_p <- vc
# transform content of documents to lower case
vc_p <- tm_map(vc_p, content_transformer(tolower))
# transform content of documents by removal of numbers
vc_p <- tm_map(vc_p, removeNumbers)
# transform content of documents by removal of punctuation
vc_p <- tm_map(vc_p, removePunctuation)
# transform content of documents by removal of stopwords
vc_p <- tm_map(vc_p, removeWords, stopwords("en"))
# remove word 'reuter' at the end of content
vc_p <- tm_map(vc_p, removeWords, c("reuter"))
# transform content of documents by removal of multispaces
vc_p <- tm_map(vc_p, stripWhitespace)

# remove unused structures
remove(allDocs, cgi, docsOfFile, files, i, ls, topic, cgiDt, lsDt, topicDt, vc, vs)

wanted_topics <- list("earn", "acq", "money-fx", "crude", "grain")
wanted <- tm_filter(vc_p, FUN = function(x){length(meta(x)[["topics_cat"]]) == 1})
wanted <- tm_filter(wanted, FUN = function(x){any(meta(x)[["topics_cat"]] %in% wanted_topics)})
wanted <- tm_filter(wanted, FUN = function(x){meta(x)[["lewissplit"]] %in% c("TRAIN", "TEST")})

dataframe <- data.frame(text=unlist(sapply(wanted, `[`, "content")),
                        topic=unlist(sapply(wanted, meta, "topics_cat")),
                        train_test=(sapply(wanted, meta, "lewissplit")),
                        stringsAsFactors=F)

sourceData <- VectorSource(dataframe$text)
corpus <- Corpus(sourceData)
dtm <- DocumentTermMatrix(corpus)
weightedDtm <- weightTfIdf(dtm)

dtm <- removeSparseTerms(dtm, 0.98)
weightedDtm <- removeSparseTerms(weightedDtm, 0.98)

dataframeDtm <- data.frame(as.matrix(dtm))
dataframeWeightedDtm <- data.frame(as.matrix(weightedDtm))

dataframeDtm_train <- dataframeDtm[which(dataframe$train_test == "TRAIN"),]
dataframeDtm_test <- dataframeDtm[which(dataframe$train_test == "TEST"),]
dataframeWeightedDtm_train <- dataframeWeightedDtm[which(dataframe$train_test == "TRAIN"),]
dataframeWeightedDtm_test <- dataframeWeightedDtm[which(dataframe$train_test == "TEST"),]

dataframeDtm_train$topic <- dataframe$topic[which(dataframe$train_test == "TRAIN")]
dataframeDtm_test$topic <- dataframe$topic[which(dataframe$train_test == "TEST")]
dataframeWeightedDtm_train$topic <- dataframe$topic[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtm_test$topic <- dataframe$topic[which(dataframe$train_test == "TEST")]


# SVM
ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3)

set.seed(100)
svm.tfidf.linear  <- train(topic ~ . , data=dataframeWeightedDtm_train, trControl = ctrl, method = "svmLinear")

set.seed(100)
svm.tfidf.radial  <- train(topic ~ . , data=dataframeWeightedDtm_train, trControl = ctrl, method = "svmRadial")

svm.tfidf.linear.predict <- predict(svm.tfidf.linear,newdata = dataframeWeightedDtm_test)
svm.tfidf.radial.predict <- predict(svm.tfidf.radial,newdata = dataframeWeightedDtm_test)

svm.tfidf.linear
svm.tfidf.radial

#SAMOSTATNY MODEL pre kazdy TOPIC, dokument ho ma alebo nema
