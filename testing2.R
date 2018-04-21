library(tm)

reuters <- VCorpus(DirSource("/home/miroslav/Desktop/SKOLA/Ing/2. semester/Objavovanie znalosti/Projekt/R/Dataset/testing5", mode = "binary"),
                   readerControl = list(reader = readReut21578XMLasPlain))

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
files <- list.files(path="Dataset/testing1/", pattern="*.sgm", full.names=T, recursive=FALSE)
allDocs <- c()
for (i in 1:length(files)){
  docsOfFile <- processFile(files[i])
  allDocs <- union(allDocs, docsOfFile)
}

# creating corpus
vs <- VectorSource(allDocs)
vc <- VCorpus(vs, readerControl = list(reader = readReut21578XMLasPlain))

vcCocoa <- tm_filter(reuters, FUN = function(x){
  if (length(meta(x)[["topics_cat"]]) == 0 || meta(x)[["topics_cat"]] == NULL){
    return(FALSE);
  } else {
    return(meta(x)[["topics_cat"]] == "cocoa")
  }
})
