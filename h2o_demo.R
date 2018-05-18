library(h2o) # for H2O Machine Learning
library(lime) # for Machine Learning Interpretation
library(mlbench) # for Datasets


# Your lucky seed here ...
n_seed = 12345


data("BostonHousing")
dim(BostonHousing)
head(BostonHousing)

# Median House Value
target = "medv" 
features = setdiff(colnames(BostonHousing), target)
print(features)

# Start a local H2O cluster (JVM)
h2o.init()
h2o.no_progress() # disable progress bar for RMarkdown
h2o.show_progress()

# H2O dataframe
h_boston = as.h2o(BostonHousing)
# Now we have an H2O dataframe
str(h_boston)

# Split Train/Test
h_split = h2o.splitFrame(h_boston, ratios = 0.75, seed = n_seed)
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




## ------------------------------------------------------------------------------

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
                          max_models = 100,         # Max no. of models
                          stopping_metric = "RMSE", # Metric to optimize
                          project_name = "my_automl",
                          exclude_algos = NULL,     # If you want to exclude any algo 
                          seed = n_seed)


model_automl@leaderboard

bestoffamily = h2o.getModel('StackedEnsemble_BestOfFamily_0_AutoML_20180518_095732')
bestoffamily

# Default GBM Model
h2o.performance(model_gbm, newdata = h_test)
# Best model from AutoML
h2o.performance(model_automl@leader, newdata = h_test) # lower RMSE = better




explainer = lime::lime(x = as.data.frame(h_train[, features]),model = model_automl@leader)

# Extract one sample (change `1` to any row you want)
d_samp = as.data.frame(h_test[1, features])
# Assign a specifc row name (for better visualization)
row.names(d_samp) = "Sample 1" 
# Create explanations
explanations = lime::explain(x = d_samp,
                             explainer = explainer,
                             n_permutations = 5000,
                             feature_select = "auto",
                             n_features = 13) # Look top x features


lime::plot_features(explanations, ncol = 1)


# Sort explanations by feature weight
explanations = 
  explanations[order(explanations$feature_weight, decreasing = TRUE),]
# Print Table
print(explanations)
