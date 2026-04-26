import zipfile
import os

zip_path = "/Users/arinbaswana/Desktop/DATA GRAPHS/cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0/CGMacros_dateshifted365.zip"
extract_dir = "/Users/arinbaswana/Desktop/DATA GRAPHS/cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0/CGMacros_dateshifted365"

print("Starting extraction...")
os.makedirs(extract_dir, exist_ok=True)
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
except Exception as e:
    print(f"Failed: {e}")
