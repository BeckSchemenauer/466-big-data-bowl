import pickle

# Load the .pkl file
with open('models/3_seconds_62.56%.pkl', 'rb') as file:  # Replace 'model.pkl' with your file path
    saved_object = pickle.load(file)

# Check the type of the loaded object
print(f"Type of loaded object: {type(saved_object)}")

# If the object is a model, check hyperparameters
if hasattr(saved_object, 'get_params'):
    # For scikit-learn models or objects with get_params method
    hyperparameters = saved_object.get_params()
    print("Hyperparameters:")
    for param, value in hyperparameters.items():
        print(f"{param}: {value}")
elif isinstance(saved_object, dict):
    # If the saved object is a dictionary, hyperparameters might be stored directly
    print("Loaded dictionary keys:")
    for key in saved_object.keys():
        print(f"{key}: {saved_object[key]}")
else:
    print("The loaded object type is not recognized or does not have hyperparameters.")
