# Libraries
library(h2o) # for H2O Machine Learning
library(lime) # for Machine Learning Interpretation
library(mlbench) # for Datasets


# Your lucky seed here ...
n_seed = 12345


data("PimaIndiansDiabetes")
dim(PimaIndiansDiabetes)
head(PimaIndiansDiabetes)

target = "diabetes" 
features = setdiff(colnames(PimaIndiansDiabetes), target)
print(features)

# Start a local H2O cluster (JVM)
h2o.init()
h2o.no_progress() # disable progress bar for RMarkdown


# H2O dataframe
h_diabetes = as.h2o(PimaIndiansDiabetes)
head(PimaIndiansDiabetes)

# Split Train/Test
h_split = h2o.splitFrame(h_diabetes, ratios = 0.75, seed = n_seed)
h_train = h_split[[1]] # 75% for modelling
h_test = h_split[[2]] # 25% for evaluation


# Train a Default H2O GBM model
model_gbm = h2o.gbm(x = features,
                    y = target,
                    training_frame = h_train,
                    model_id = "my_gbm",
                    seed = n_seed)
print(model_gbm)


# Evaluate performance on test
h2o.performance(model_gbm, newdata = h_test)

# Train multiple H2O models with H2O AutoML
# Stacked Ensembles will be created from those H2O models
# You tell H2O ...
#     1) how much time you have and/or 
#     2) how many models do you want
# Note: H2O deep learning algo on multi-core is stochastic
model_automl = h2o.automl(x = features,
                          y = target,
                          training_frame = h_train,
                          nfolds = 5,               # Cross-Validation
                          max_runtime_secs = 120,   # Max time
                          max_models = 10,         # Max no. of models
                          stopping_metric = "AUC", # Metric to optimize
                          project_name = "my_automl",
                          exclude_algos = NULL,     # If you want to exclude any algo 
                          seed = n_seed)


