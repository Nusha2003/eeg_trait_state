import joblib
import pprint

# Load the results from the file
results = joblib.load("/home1/amadapur/projects/eeg_trait_state_geometry/permutation_across_spaces.pkl")

# Use pprint (pretty print) to make the dictionary readable
print("Permutation results")
pprint.pprint(results)