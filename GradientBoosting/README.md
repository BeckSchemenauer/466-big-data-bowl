Run feature_selection_preprocessing.py to populate AfterSnap

xg_boost_tuning.py is a file for discovering optimal xgboost parameters, it does not contain every parameter tested and is constantly changing.

xg_boost.py contains the final running of the top implementation with cross validation

models is a directory containing the top models produced by xgboost. Currently the github only contains the top model on a 10 yard boundary 3 seconds after snap.