library(tm)
library(ggplot2)

# function for processing documents inside file
processFile <- function(filepath) {
  con = file(filepath, "r")
  resultVector <- c()
  x <- character()The.Walking.Dead.S08E16.HDTV.x264-FLEET
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
vc_p <- tm_map(vc_p, removeNumbers)The.Walking.Dead.S08E16.HDTV.x264-FLEET
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

# get subset corpus of train documents
vc_p1000 <- tm_filter(vc_p, FUN = function(x){meta(x)[["lewissplit"]] == "TRAIN"})[1:1000]

# creating Term Document Matrix
tdm <- TermDocumentMatrix(vc_p1000)[, 1:10]
dim(tdm)

# creating matrix
tdmMatrix <- as.matrix(tdm)
tdmMatrix <- cbind.data.frame(tdmMatrix, train_test=rep("train", nrow(tdmMatrix)))
freq <- sort(rowSums(tdmMatrix), decreasing=TRUE)
wf <- data.frame(word=names(freq), freq=freq)

# construct graph of term frequencies
p <- ggplot(subset(wf, freq>100), aes(x = reorder(word, -freq), y = freq)) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x=element_text(angle=45, hjust=1))
p

findFreqTerms(tdm, 50)
findAssocs(tdm, "said", 0.7)
tdm_p <- removeSparseTerms(tdm, 0.8)
inspect(tdm_p)

#library(reshape2)
#TDM.dense = melt(tdm.dense, value.name = "count")
#head(tdm.dense)

weightedtdm <- weightTfIdf(tdm)

as.matrix(weightedtdm)[10:20,200:210]
















# load the data
r8train <- read.table("r8-train-all-terms.txt", header=FALSE, sep='\t')
r8test <- read.table("r8-test-all-terms.txt", header=FALSE, sep='\t')

# explore the structure of the  data
str(r8train)
str(r8test)

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
corpus[[20]]$content

corpus <- tm_map(corpus, content_transformer(tolower)) # convert to lowercase
corpus <- tm_map(corpus, removeNumbers) # remove digits
corpus <- tm_map(corpus, removePunctuation) # remove punctuation
corpus <- tm_map(corpus, stripWhitespace) # strip extra whitespace
corpus <- tm_map(corpus, removeWords, stopwords('english')) # remove stopwords

corpus[[20]]$content

tdm <- DocumentTermMatrix(corpus)

library(tm)

reut21578 <- system.file("texts", "crude", package = "tm")
reuters <- VCorpus(DirSource(reut21578, mode = "binary"), 
                     readerControl = list(reader = readReut21578XMLasPlain))

reuters <- VCorpus(DirSource("Dataset/reuters21578/test", mode = "binary"), 
                     readerControl = list(reader = readReut21578XMLasPlain))

cars <- c("FORD", "GM")
price  <- list( c(1000, 2000, 3000),  c(2000, 500, 1000))
myDF <- data.frame(cars=cars, price=cbind(price))

processFile = function(filepath) {
  con = file(filepath, "r")
  x <- character()
  while ( TRUE ) {
    line = readLines(con, n = 1)
    line = gsub("&#\\d+;", "", line)
    if ( length(line) == 0){
      break
    }
    x = paste(x, line)
  }
  
  close(con)
  return(x)
}

library(tm)
x <- processFile("Dataset/reuters21578/test/reut-00001.xml")
print(x)
bla <- VectorSource(x)
bla2 <- VCorpus(bla, readerControl = list(reader = readReut21578XML))
