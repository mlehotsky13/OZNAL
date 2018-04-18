library(tm)

r8train <- read.table("r8-train-all-terms.txt", header=FALSE, sep='\t')
r8test <- read.table("r8-test-all-terms.txt", header=FALSE, sep='\t')

names(r8train) <- c("Class", "docText")
names(r8test) <- c("Class", "docText")

r8train$docText <- as.character(r8train$docText)
r8test$docText <- as.character(r8test$docText)

r8train$train_test <- c("train")
r8test$train_test <- c("test")

merged <- rbind(r8train, r8test)

remove(r8train, r8test)

merged <- merged[which(merged$Class %in% c("crude","money-fx","trade")),]

merged$Class <- droplevels(merged$Class)

table(merged$Class,merged$train_test)

sourceData <- VectorSource(merged$docText)

corpus <- Corpus(sourceData)

corpus <- tm_map(corpus, content_transformer(tolower)) # convert to lowercase
corpus <- tm_map(corpus, removeNumbers) # remove digits
corpus <- tm_map(corpus, removePunctuation) # remove punctuation
corpus <- tm_map(corpus, stripWhitespace) # strip extra whitespace
corpus <- tm_map(corpus, removeWords, stopwords('english')) # remove stopwords

tdm <- DocumentTermMatrix(corpus)
weightedtdm <- weightTfIdf(tdm)

tdm <- as.data.frame(inspect(tdm))
weightedtdm <- as.data.frame(inspect(weightedtdm))

tdmTrain <- tdm[which(merged$train_test == "train"),]
weightedTDMtrain <- weightedtdm[which(merged$train_test == "train"),]

tdmTest <-  tdm[which(merged$train_test == "test"),]
weightedTDMtest <- weightedtdm[which(merged$train_test == "test"),]

tdmTrain$doc.class <- merged$Class[which(merged$train_test == "train")]
tdmTest$doc.class <- merged$Class[which(merged$train_test == "test")]
weightedTDMtrain$doc.class <- merged$Class[which(merged$train_test == "train")]
weightedTDMtest$doc.class  <- merged$Class[which(merged$train_test == "test")]
