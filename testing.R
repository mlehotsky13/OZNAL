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
                    topics_cat = as.list(meta(x, "topics_cat")),
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
wantedTopics <- list("earn", "acq", "money-fx", "crude", "grain", "trade", "interest", "wheat", "ship", "corn")
wanted <- tm_filter(vcProcessed, FUN = function(x){length(meta(x)[["topics_cat"]]) > 0})
wanted <- tm_filter(wanted, FUN = function(x){all(meta(x)[["topics_cat"]] %in% wantedTopics)})
wanted <- tm_filter(wanted, FUN = function(x){meta(x)[["lewissplit"]] %in% c("TRAIN", "TEST")})

# create dataframe with text contents, topic and indication of presence in train or test set
dataframe <- data.frame(text=unlist(sapply(wanted, `[`, "content")),
                        topic=unlist(sapply(wanted, meta, "topics_cat")),
                        train_test=(sapply(wanted, meta, "lewissplit")),
                        stringsAsFactors=F)

# create dataframe with text contents, 10 topic specific columns and indication of presence in train or test set
dataframe <- data.frame(text=unlist(sapply(wanted, `[`, "content")),
                        earn=as.factor(unlist(sapply(wanted, FUN = function(x){("earn" %in% meta(x)[["topics_cat"]]) * 1}))),
                        acq=as.factor(unlist(sapply(wanted, FUN = function(x){("acq" %in% meta(x)[["topics_cat"]]) * 1}))),
                        moneyfx=as.factor(unlist(sapply(wanted, FUN = function(x){"money-fx" %in% meta(x)[["topics_cat"]] * 1}))),
                        crude=as.factor(unlist(sapply(wanted, FUN = function(x){"crude" %in% meta(x)[["topics_cat"]] * 1}))),
                        grain=as.factor(unlist(sapply(wanted, FUN = function(x){"grain" %in% meta(x)[["topics_cat"]] * 1}))),
                        trade=as.factor(unlist(sapply(wanted, FUN = function(x){"trade" %in% meta(x)[["topics_cat"]] * 1}))),
                        interest=as.factor(unlist(sapply(wanted, FUN = function(x){"interest" %in% meta(x)[["topics_cat"]] * 1}))),
                        wheat=as.factor(unlist(sapply(wanted, FUN = function(x){"wheat" %in% meta(x)[["topics_cat"]] * 1}))),
                        ship=as.factor(unlist(sapply(wanted, FUN = function(x){"ship" %in% meta(x)[["topics_cat"]] * 1}))),
                        corn=as.factor(unlist(sapply(wanted, FUN = function(x){"corn" %in% meta(x)[["topics_cat"]] * 1}))),
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

# set 10 topic columns for dataframeDtmTrain
dataframeDtmTrain$earn <- dataframe$earn[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$acq <- dataframe$acq[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$moneyfx <- dataframe$moneyfx[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$crude <- dataframe$crude[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$grain <- dataframe$grain[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$trade <- dataframe$trade[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$interest <- dataframe$interest[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$wheat <- dataframe$wheat[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$ship <- dataframe$ship[which(dataframe$train_test == "TRAIN")]
dataframeDtmTrain$corn <- dataframe$corn[which(dataframe$train_test == "TRAIN")]

# set 10 topic columns for dataframeDtmTest
dataframeDtmTest$earn <- dataframe$earn[which(dataframe$train_test == "TEST")]
dataframeDtmTest$acq <- dataframe$acq[which(dataframe$train_test == "TEST")]
dataframeDtmTest$moneyfx <- dataframe$moneyfx[which(dataframe$train_test == "TEST")]
dataframeDtmTest$crude <- dataframe$crude[which(dataframe$train_test == "TEST")]
dataframeDtmTest$grain <- dataframe$grain[which(dataframe$train_test == "TEST")]
dataframeDtmTest$trade <- dataframe$trade[which(dataframe$train_test == "TEST")]
dataframeDtmTest$interest <- dataframe$interest[which(dataframe$train_test == "TEST")]
dataframeDtmTest$wheat <- dataframe$wheat[which(dataframe$train_test == "TEST")]
dataframeDtmTest$ship <- dataframe$ship[which(dataframe$train_test == "TEST")]
dataframeDtmTest$corn <- dataframe$corn[which(dataframe$train_test == "TEST")]

# set 10 topic columns for dataframeDtmTrain
dataframeWeightedDtmTrain$earn <- dataframe$earn[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$acq <- dataframe$acq[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$moneyfx <- dataframe$moneyfx[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$crude <- dataframe$crude[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$grain <- dataframe$grain[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$trade <- dataframe$trade[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$interest <- dataframe$interest[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$wheat <- dataframe$wheat[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$ship <- dataframe$ship[which(dataframe$train_test == "TRAIN")]
dataframeWeightedDtmTrain$corn <- dataframe$corn[which(dataframe$train_test == "TRAIN")]

# set 10 topic columns for dataframeWeightedDtmTest
dataframeWeightedDtmTest$earn <- dataframe$earn[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$acq <- dataframe$acq[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$moneyfx <- dataframe$moneyfx[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$crude <- dataframe$crude[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$grain <- dataframe$grain[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$trade <- dataframe$trade[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$interest <- dataframe$interest[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$wheat <- dataframe$wheat[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$ship <- dataframe$ship[which(dataframe$train_test == "TEST")]
dataframeWeightedDtmTest$corn <- dataframe$corn[which(dataframe$train_test == "TEST")]

# select subset of dataframes
dataframeWeightedDtmTrain <- dataframeWeightedDtmTrain[1:200,]
dataframeWeightedDtmTest <- dataframeWeightedDtmTest[1:80,]
dataframeDtmTrain <- dataframeDtmTrain[1:200,]
dataframeDtmTest <- dataframeDtmTest[1:80,]

# function for creating data frame from predictions
createResultsDF <- function(predictions){
  
  df <- data.frame(Topic=character(),
                   Accuracy=double(),
                   Precision=double(),
                   Recall=double(),
                   F1=double())
  
  for (i in 1:length(predictions)){
    cm <- confusionMatrix(predictions[[i]], dataframeDtmTest[[names(predictions)[[i]]]])
    d <- data.frame(Topic=names(predictions)[[i]],
                    Accuracy=cm$overall[["Accuracy"]], 
                    Precision=precision(as.table(cm)),
                    Recall=recall(as.table(cm)), 
                    F1=F_meas(as.table(cm)));
    
    df <- rbind(df, d);
  }
  
  return(df);
}

#KNN
ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3)

# for topic column
knnTf <- train(topic ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfidf <- train(topic ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)

knnPredict <- predict(knnTf, newdata = dataframeDtmTest)
knnTfidfPredict <- predict(knnTfidf, newdata = dataframeWeightedDtmTest)

knnTfResults1 <- createResultsDF(as.list(knnPredict));
knnTfidfResults1 <- createResultsDF(as.list(knnTfidfPredict));
  
# for 10 specific topic columns
knnTfEarn <- train(earn ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfAcq <- train(acq ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfMoneyfx <- train(moneyfx ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfCrude <- train(crude ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfGrain <- train(grain ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfTrade <- train(trade ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfInterest <- train(interest ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfWheat <- train(wheat ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfShip <- train(ship ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)
knnTfCorn <- train(corn ~ ., data = dataframeDtmTrain, method = "knn", trControl = ctrl)

knnTfidfEarn <- train(earn ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfAcq <- train(acq ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfMoneyfx <- train(moneyfx ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfCrude <- train(crude ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfGrain <- train(grain ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfTrade <- train(trade ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfInterest <- train(interest ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfWheat <- train(wheat ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfShip <- train(ship ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)
knnTfidfCorn <- train(corn ~ ., data = dataframeWeightedDtmTrain, method = "knn", trControl = ctrl)

knnTfPredictions = list()
knnTfPredictions[["earn"]] <- predict(knnTfEarn, newdata = dataframeDtmTest)
knnTfPredictions[["acq"]] <- predict(knnTfAcq, newdata = dataframeDtmTest)
knnTfPredictions[["moneyfx"]] <- predict(knnTfMoneyfx, newdata = dataframeDtmTest)
knnTfPredictions[["crude"]] <- predict(knnTfCrude, newdata = dataframeDtmTest)
knnTfPredictions[["grain"]] <- predict(knnTfGrain, newdata = dataframeDtmTest)
knnTfPredictions[["trade"]] <- predict(knnTfTrade, newdata = dataframeDtmTest)
knnTfPredictions[["interest"]] <- predict(knnTfInterest, newdata = dataframeDtmTest)
knnTfPredictions[["wheat"]] <- predict(knnTfWheat, newdata = dataframeDtmTest)
knnTfPredictions[["ship"]] <- predict(knnTfShip, newdata = dataframeDtmTest)
knnTfPredictions[["corn"]] <- predict(knnTfCorn, newdata = dataframeDtmTest)

knnTfResults2 <- createResultsDF(knnTfPredictions);

knnTfidfPredictions = list()
knnTfidfPredictions[["earn"]] <- predict(knnTfidfEarn, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["acq"]] <- predict(knnTfidfAcq, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["moneyfx"]] <- predict(knnTfidfMoneyfx, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["crude"]] <- predict(knnTfidfCrude, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["grain"]] <- predict(knnTfidfGrain, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["trade"]] <- predict(knnTfidfTrade, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["interest"]] <- predict(knnTfidfInterest, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["wheat"]] <- predict(knnTfidfWheat, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["ship"]] <- predict(knnTfidfShip, newdata = dataframeWeightedDtmTest)
knnTfidfPredictions[["corn"]] <- predict(knnTfidfCorn, newdata = dataframeWeightedDtmTest)

knnTfidfResults2 <- createResultsDF(knnTfidfPredictions);

# SVM
ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3)

# for topic column
svmTfLinear  <- train(topic ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinear  <- train(topic ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")

svmTfRadial  <- train(topic ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmRadial")
svmTfidfRadial  <- train(topic ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmRadial")

svmTfLinearPredict <- predict(svmTfLinear, newdata = dataframeDtmTest)
svmTfidfLinearPredict <- predict(svmTfidfLinear, newdata = dataframeWeightedDtmTest)
svmTfRadialPredict <- predict(svmTfRadial, newdata = dataframeDtmTest)
svmTfidfRadialPredict <- predict(svmTfidfRadial, newdata = dataframeWeightedDtmTest)

svmTfLinearResults1 <- createResultsDF(as.list(svmTfLinearPredict));
svmTfidfLinearResults1 <- createResultsDF(as.list(svmTfidfLinearPredict));
svmTfRadialResults <- createResultsDF(as.list(svmTfRadialPredict));
svmTfidfRadialResults <- createResultsDF(as.list(svmTfidfRadialPredict));

# for 10 specific topic columns
svmTfLinearEarn  <- train(earn ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearAcq  <- train(acq ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearMoneyfx  <- train(moneyfx ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearCrude  <- train(crude ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearGrain  <- train(grain ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearTrade  <- train(trade ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearInterest  <- train(interest ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearWheat  <- train(wheat ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearShip  <- train(ship ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfLinearCorn  <- train(corn ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "svmLinear")

svmTfidfLinearEarn  <- train(earn ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearAcq  <- train(acq ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearMoneyfx  <- train(moneyfx ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearCrude  <- train(crude ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearGrain  <- train(grain ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearTrade  <- train(trade ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearInterest  <- train(interest ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearWheat  <- train(wheat ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearShip  <- train(ship ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")
svmTfidfLinearCorn  <- train(corn ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "svmLinear")

svmTfPredictions = list()
svmTfPredictions[["earn"]] <- predict(svmTfLinearEarn, newdata = dataframeDtmTest)
svmTfPredictions[["acq"]] <- predict(svmTfLinearAcq, newdata = dataframeDtmTest)
svmTfPredictions[["moneyfx"]] <- predict(svmTfLinearMoneyfx, newdata = dataframeDtmTest)
svmTfPredictions[["crude"]] <- predict(svmTfLinearCrude, newdata = dataframeDtmTest)
svmTfPredictions[["grain"]] <- predict(svmTfLinearGrain, newdata = dataframeDtmTest)
svmTfPredictions[["trade"]] <- predict(svmTfLinearTrade, newdata = dataframeDtmTest)
svmTfPredictions[["interest"]] <- predict(svmTfLinearInterest, newdata = dataframeDtmTest)
svmTfPredictions[["wheat"]] <- predict(svmTfLinearWheat, newdata = dataframeDtmTest)
svmTfPredictions[["ship"]] <- predict(svmTfLinearShip, newdata = dataframeDtmTest)
svmTfPredictions[["corn"]] <- predict(svmTfLinearCorn, newdata = dataframeDtmTest)

svmTfLinearResults2 <- createResultsDF(svmTfPredictions);

svmTfidfPredictions = list()
svmTfidfPredictions[["earn"]] <- predict(svmTfidfLinearEarn, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["acq"]] <- predict(svmTfidfLinearAcq, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["moneyfx"]] <- predict(svmTfidfLinearMoneyfx, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["crude"]] <- predict(svmTfidfLinearCrude, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["grain"]] <- predict(svmTfidfLinearGrain, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["trade"]] <- predict(svmTfidfLinearTrade, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["interest"]] <- predict(svmTfidfLinearInterest, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["wheat"]] <- predict(svmTfidfLinearWheat, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["ship"]] <- predict(svmTfidfLinearShip, newdata = dataframeWeightedDtmTest)
svmTfidfPredictions[["corn"]] <- predict(svmTfidfLinearCorn, newdata = dataframeWeightedDtmTest)

svmTfidfLinearResults2 <- createResultsDF(svmTfidfPredictions);

# Decision Tree
ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3)

# for topic column
treeTf  <- train(topic ~ . , data = dataframeDtmTrain, method = "rpart", trControl = ctrl )
treeTfidf  <- train(topic ~ . , data = dataframeWeightedDtmTrain, method = "rpart", trControl = ctrl )

treeTfPredict <- predict(treeTf, newdata = dataframeDtmTest)
treeTfidfPredict <- predict(treeTfidf, newdata = dataframeWeightedDtmTest)

treeTfResults1 <- createResultsDF(treeTfPredict);
treeTfidfResults1 <- createResultsDF(treeTfidfPredict);

# for 10 specific topic columns
treeTfEarn  <- train(earn ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfAcq  <- train(acq ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfMoneyfx  <- train(moneyfx ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfCrude  <- train(crude ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfGrain  <- train(grain ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfTrade  <- train(trade ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfInterest  <- train(interest ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfWheat  <- train(wheat ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfShip  <- train(ship ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")
treeTfCorn  <- train(corn ~ . , data=dataframeDtmTrain, trControl = ctrl, method = "rpart")

treeTfidfEarn  <- train(earn ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfAcq  <- train(acq ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfMoneyfx  <- train(moneyfx ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfCrude  <- train(crude ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfGrain  <- train(grain ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfTrade  <- train(trade ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfInterest  <- train(interest ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfWheat  <- train(wheat ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfShip  <- train(ship ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")
treeTfidfCorn  <- train(corn ~ . , data=dataframeWeightedDtmTrain, trControl = ctrl, method = "rpart")

treeTfPredictions = list()
treeTfPredictions[["earn"]] <- predict(knnTfEarn, newdata = dataframeDtmTest)
treeTfPredictions[["acq"]] <- predict(knnTfAcq, newdata = dataframeDtmTest)
treeTfPredictions[["moneyfx"]] <- predict(knnTfMoneyfx, newdata = dataframeDtmTest)
treeTfPredictions[["crude"]] <- predict(knnTfCrude, newdata = dataframeDtmTest)
treeTfPredictions[["grain"]] <- predict(knnTfGrain, newdata = dataframeDtmTest)
treeTfPredictions[["trade"]] <- predict(knnTfTrade, newdata = dataframeDtmTest)
treeTfPredictions[["interest"]] <- predict(knnTfInterest, newdata = dataframeDtmTest)
treeTfPredictions[["wheat"]] <- predict(knnTfWheat, newdata = dataframeDtmTest)
treeTfPredictions[["ship"]] <- predict(knnTfShip, newdata = dataframeDtmTest)
treeTfPredictions[["corn"]] <- predict(knnTfCorn, newdata = dataframeDtmTest)

treeTfResults2 <- createResultsDF(treeTfPredictions);

treeTfidfPredictions = list()
treeTfidfPredictions[["earn"]] <- predict(knnTfEarn, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["acq"]] <- predict(knnTfAcq, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["moneyfx"]] <- predict(knnTfMoneyfx, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["crude"]] <- predict(knnTfCrude, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["grain"]] <- predict(knnTfGrain, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["trade"]] <- predict(knnTfTrade, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["interest"]] <- predict(knnTfInterest, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["wheat"]] <- predict(knnTfWheat, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["ship"]] <- predict(knnTfShip, newdata = dataframeWeightedDtmTest)
treeTfidfPredictions[["corn"]] <- predict(knnTfCorn, newdata = dataframeWeightedDtmTest)

treeTfidfResults2 <- createResultsDF(treeTfidfPredictions);

# Neural Network
ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3)

# for topic column
neuralNetTf  <- train(topic ~ . , data = dataframeDtmTrain, method = "dnn", trControl = ctrl )
neuralNetTfidf  <- train(topic ~ . , data = dataframeWeightedDtmTrain, method = "dnn", trControl = ctrl )

neuralNetTfPredict <- predict(neuralNetTf, newdata = dataframeDtmTest)
neuralNetTfidfPredict <- predict(neuralNetTfidf, newdata = dataframeWeightedDtmTest)

neuralNetTfResults1 <- createResultsDF(neuralNetTfPredict)
neuralNetTfidfResults1 <- createResultsDF(neuralNetTfidfPredict)