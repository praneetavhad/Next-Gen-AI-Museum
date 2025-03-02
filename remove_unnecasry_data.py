input_file = "train.txt"  # Replace with your actual filename
output_file = "train2.txt"  # The filtered output file

# Read the file and filter lines
with open(input_file, "r") as f:
    lines = f.readlines()

# Keep lines that contain "museum-indoor" or "museum-outdoor"
filtered_lines = [line.strip() for line in lines if "museum-indoor" in line or "museum-outdoor" in line]

# Write the filtered lines to the output file
with open(output_file, "w") as f:
    for line in filtered_lines:
        f.write(line + "\n")

print("Filtering complete. Check", output_file)