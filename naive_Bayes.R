setwd("~/repos/kaggle-personalized-med")
library(methods) # Load if running in terminal with Rscript command (weird inconsistency in loading packages)

####### Tidy
rm(list = ls())

# Training variants
training_variants <- read.csv("data/training_variants")
# Test variants
test_variants <- read.csv("data/test_variants")

# Create training text tibble
g <- do.call(rbind,strsplit(readLines('data/training_text'),"||",fixed=T))
library(dplyr)
training_text <- data_frame(ID = g[-1,1], Text = g[-1,2])
training_text$ID <- as.numeric(training_text$ID)
training_text <- left_join(training_text, training_variants)
# Create test text tibble
h <- do.call(rbind,strsplit(readLines('data/test_text'),"||",fixed=T))
test_text <- data_frame(ID = h[-1,1], Text = h[-1,2])
test_text$ID <- as.numeric(test_text$ID)

# Load common words
library(tidytext)
data(stop_words)

# Unnest tokens, documents with counts for each word for TRAINING
doc_words <- training_text %>%
  unnest_tokens(word, Text) %>%
  filter(!word %in% stop_words$word) %>% # remove common words
  count(ID, word, sort=FALSE) %>%
  ungroup() %>%
  left_join(training_variants)
# Remove numbers
doc_words <- doc_words[is.na(as.numeric(doc_words$word)),]

# Unnest tokens, documents with counts for each word for TEST
doc_words_test <- test_text %>%
  unnest_tokens(word, Text) %>%
  filter(!word %in% stop_words$word) %>% # remove common words
  count(ID, word, sort=FALSE) %>%
  ungroup() %>%
  left_join(test_variants)
# Remove numbers
doc_words_test <- doc_words_test[is.na(as.numeric(doc_words_test$word)),]

# Create Document Term Matrix for training
library(tm)
doc_dtm <- doc_words %>%
  cast_dtm(ID, word, n)
# Remove sparse terms
doc_dtm <- removeSparseTerms(doc_dtm, 0.9)

# Create Document Term Matrix for test
doc_dtm_test <- doc_words_test %>%
  cast_dtm(ID, word, n)
# Remove sparse terms
doc_dtm_test <- removeSparseTerms(doc_dtm_test, 0.9)

####### Supervised Analysis with naive Bayes (using just training data to measure accuracy)
set.seed(1234)

# Get the term frequency-inverse document frequency (tf-idf)
tf_weight <- weightTfIdf(doc_dtm)
tf_weight <- as.matrix(tf_weight)

# Random sample
rand <- sample(1:dim(tf_weight)[1], dim(tf_weight)[1], replace = FALSE)
rand_test <- rand[1:(length(rand)/5)]
rand_train <- rand[(length(rand_test)+1):length(rand)]

# Get target classes
target_class <- training_variants$Class[training_variants$ID %in% as.numeric(dimnames(tf_weight)$Docs)]

# Create model
library(e1071)
classifier <- naiveBayes(x = tf_weight[rand_train,], y = target_class[rand_train])

# Predict, and get the posterior probabilities for each class
pred_class_prob <- predict(classifier, tf_weight[rand_test,], type = "raw")

# Find accuracy, taking class as highest probability
pred_class <- apply(pred_class_prob, 1, which.max)
cont_table <- table(pred_class, target_class[rand_test])
sum(diag(cont_table))/sum(cont_table)

# Is it better than random? It could be..
set.seed(555)
rand <- sample(1:dim(tf_weight)[1], dim(tf_weight)[1], replace = FALSE)
rand_test <- rand[1:(length(rand)/5)]
cont_table <- table(pred_class, target_class[rand_test])
sum(diag(cont_table))/sum(cont_table)

####### Supervised Analysis with naive Bayes to predict test data
# Get the term frequency-inverse document frequency for test (tf-idf)
tf_weight_test <- weightTfIdf(doc_dtm_test)
tf_weight_test <- as.matrix(tf_weight_test)

# Random sample
set.seed(1234)
rand <- sample(1:dim(tf_weight)[1], dim(tf_weight)[1], replace = FALSE)

# Build model with all training and predict test
classifier <- naiveBayes(x = tf_weight[rand,], y = target_class[rand])
pred_class_prob <- predict(classifier, tf_weight_test, type = "raw")

# Label data ready for submission
test_ids <- as.numeric(dimnames(tf_weight_test)$Docs)
pred_class_prob <- cbind(test_ids, pred_class_prob)
colnames(pred_class_prob) <- c("ID", "class1", "class2", "class3", "class4", "class5",
                               "class6", "class7", "class8", "class9")

# Save important objects
save(classifier, pred_class_prob, file = "naiveBayes.RData")

# Write submission file
write.csv(pred_class_prob, "Submission_File.csv", row.names = FALSE)
