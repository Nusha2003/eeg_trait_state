import joblib
import pprint

# Load the results from the file
results = joblib.load("/home1/amadapur/projects/eeg_trait_state_geometry/full_space_stress_test.pkl")

# Use pprint (pretty print) to make the dictionary readable
print("--- EEG Decoding Stress Test Results ---")
pprint.pprint(results)