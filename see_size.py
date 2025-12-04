import numpy as np
import os

# Define the names of the .npy files
file_names = ['lungs.npy', 'ws.npy', 'croped.npy']

# Get the path to the user's desktop
# This path works for most Linux and macOS systems.
# For Windows, you might need to adjust it, e.g., 'C:/Users/YourUsername/Desktop'
desktop_path = os.path.expanduser('~/Desktop')

## Check the Size of .npy Images 💾

print(f"Checking .npy file sizes in: **{desktop_path}**\n")

for name in file_names:
    file_path = os.path.join(desktop_path, name)
    
    # Check if the file exists
    if os.path.exists(file_path):
        try:
            # Load the NumPy array from the file
            data = np.load(file_path)
            
            # Print the file name and the shape (size) of the array
            print(f"**{name}**")
            print(f"  - Shape (Dimensions): {data.shape}")
            print(f"  - Data Type (dtype): {data.dtype}\n")
            
        except ValueError as e:
            # Handle files that might not be valid NumPy arrays
            print(f"Error loading **{name}**: Invalid NumPy file. Details: {e}\n")
            
        except Exception as e:
            # Handle other potential errors (e.g., file corruption)
            print(f"An unexpected error occurred with **{name}**: {e}\n")
            
    else:
        # File not found
        print(f"❌ **{name}** was not found in **{desktop_path}**.\n")

print("---")
print("Script finished.")