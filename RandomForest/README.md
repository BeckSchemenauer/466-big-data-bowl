Run feature_selection_preprocessing.py to populate AfterSnap

Run feature_selection_eda.py to generate graphs of after_snap data. Currently uses after_snap_5

rf_helper.py contains preprocessing and report generation used by random forest and XGBoost.

random_forest_tuning.py is a file for discovering optimal random forest parameters, it does not contain every parameter tested and is constantly changing.

random_forest.py contains the final running of the top implementation with cross validation

models is a directory containing the top models produced by random forest. Currently the github only contains the top model on a 10 yard boundary 3 seconds after snap.