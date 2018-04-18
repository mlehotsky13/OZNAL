library(tm)
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

# get subset corpus of train documents
#vc_p1000 <- tm_filter(vc_p, FUN = function(x){meta(x)[["lewissplit"]] == "TRAIN"})

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

