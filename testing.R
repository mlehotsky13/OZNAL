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

wanted <- tm_filter(vc_p, FUN = function(x){length(meta(x)[["topics_cat"]]) == 1})
wanted <- tm_filter(wanted, FUN = function(x){
  "crude" == meta(x)[["topics_cat"]] | "money-fx" == meta(x)[["topics_cat"]] | "trade" == meta(x)[["topics_cat"]]
})
wanted <- tm_filter(wanted, FUN = function(x){meta(x)[["lewissplit"]] %in% c("TRAIN", "TEST")})

#wanted_train <- tm_filter(wanted, FUN = function(x){meta(x)[["lewissplit"]] == "TRAIN"})
#wanted_test <- tm_filter(wanted, FUN = function(x){meta(x)[["lewissplit"]] == "TEST"})

dataframe <- data.frame(text=unlist(sapply(wanted, `[`, "content")),
                        topic=unlist(sapply(wanted, meta, "topics_cat")),
                        train_test=(sapply(wanted, meta, "lewissplit")),
                        stringsAsFactors=F)

sourceData <- VectorSource(dataframe$text)
corpus <- Corpus(sourceData)
dtm <- DocumentTermMatrix(corpus)
weightedDtm <- weightTfIdf(dtm)

dtm <- removeSparseTerms(dtm, 0.9)
weightedDtm <- removeSparseTerms(weightedDtm, 0.9)

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

svm.tfidf.linear.predict <- predict(svm.tfidf.linear,newdata = dataframeWeightedDtm_test)
svm.tfidf.radial.predict <- predict(svm.tfidf.radial,newdata = dataframeWeightedDtm_test)

svm.tfidf.linear
svm.tfidf.radial

#dataframe2 <- data.frame(as.matrix(weightedDtm))
#dataframe2$topic <- dataframe$topic

ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3)

set.seed(100)
svm.tfidf.linear  <- train(topic ~ . , data=dataframeWeightedDtm_train, trControl = ctrl, method = "svmLinear")

set.seed(100)
svm.tfidf.radial  <- train(topic ~ . , data=dataframeWeightedDtm_train, trControl = ctrl, method = "svmRadial")

# get subset corpus of train documents
vc_p_train <- tm_filter(vc_p, FUN = function(x){meta(x)[["lewissplit"]] == "TRAIN"})
vc_p_test <- tm_filter(vc_p, FUN = function(x){meta(x)[["lewissplit"]] == "TEST"})

vc_p_train_1 <- tm_filter(vc_p_train, FUN = function(x){length(meta(x)[["topics_cat"]]) == 1})
vc_p_train_1_non_empty <- tm_filter(vc_p_train_1, FUN = function(x){length(x$content) != 0})

vc_p_train_3 <- tm_filter(vc_p_train_1_non_empty, FUN = function(x){
  length(meta(x)[["topics_cat"]]) == 1 & "crude" %in% meta(x)[["topics_cat"]]  | "money-fx" == meta(x)[["topics_cat"]] | "trade" == meta(x)[["topics_cat"]]
})

dataframe <- data.frame(text=unlist(sapply(vc_p_train_3[1:100], `[`, "content")),
                        topic=unlist(sapply(vc_p_train_3[1:100], meta, "topics_cat")),
                        stringsAsFactors=F)


# vytvorenie corpusu obsahujuceho dokumenty s topicom "earn" => pozitivne priklad
# a dokumentov 
#vc_p_train_earn <- tm_filter(vc_p_train, FUN = function(x){
#  "earn" %in% meta(x)[["topics_cat"]] | length(meta(x)[["topics_cat"]]) == 0
#});

vc_p_one <- tm_filter(vc_p_train, FUN = function(x){length(meta(x)[["topics_cat"]]) == 1})

dataframe <- data.frame(text=unlist(sapply(vc_p_one[1:10], `[`, "content")),
                        #train_test=sapply(vc_p_train[1:10], meta, "lewissplit"),
                        topics=unlist(sapply(vc_p_one[1:10], function(x){
                          list(meta(x)[["topics_cat"]])
                        })),
                        stringsAsFactors=F)

dataframe <- data.frame(topics_cat=lapply(vc_p_train[1:10], FUN = function(x){
  if (length(meta(x)[["topics_cat"]]) == 0){
    return("NA");
  } else {
    return(meta(x)[["topics_cat"]]);
  }
}))

#text=lapply(unlist(lapply(sapply(corpus, '[', "content"),paste,collapse="\n")),
#            stringsAsFactors=FALSE))

# create Term Document Matrices
tdm_train <- TermDocumentMatrix(vc_p_train)
tdm_tfidf_train <- weightTfIdf(tdm_train)
tdm_test <- TermDocumentMatrix(vc_p_test)
tdm_tfidf_test <- weightTfIdf(tdm_test)

# create data frames
df_train <- as.data.frame(inspect(tdm_train))
df_tfidf_train <- as.data.frame(inspect(tdm_tfidf_train))
df_test <- as.data.frame(inspect(tdm_test))
df_tfidf_test <- as.data.frame(inspect(tdm_tfidf_test))

# creating Term Document Matrix
#tdm <- TermDocumentMatrix(vc_p1000)[, 1:10]
#dim(tdm)

# creating matrix
#tdmMatrix <- as.matrix(tdm)
#tdmMatrix <- cbind.data.frame(tdmMatrix, train_test=rep("train", nrow(tdmMatrix)))
#freq <- sort(rowSums(tdmMatrix), decreasing=TRUE)
#wf <- data.frame(word=names(freq), freq=freq)

# construct graph of term frequencies
#p <- ggplot(subset(wf, freq>100), aes(x = reorder(word, -freq), y = freq)) +
#  geom_bar(stat = "identity") + 
#  theme(axis.text.x=element_text(angle=45, hjust=1))
#p

#findFreqTerms(tdm, 50)
#findAssocs(tdm, "said", 0.7)
#tdm_p <- removeSparseTerms(tdm, 0.8)
#inspect(tdm_p)

#library(reshape2)
#TDM.dense = melt(tdm.dense, value.name = "count")
#head(tdm.dense)

#weightedtdm <- weightTfIdf(tdm)

#as.matrix(weightedtdm)[10:20,200:210]

#SAMOSTATNY MODEL pre kazdy TOPIC, dokument ho ma alebo nema
