from collections import Counter

# Specify the file path
file_path = '/home/xiruij/anticipation/datasets/clap/test.txt'
output_path = '/home/xiruij/anticipation/datasets/clap/test_unique.txt'

# Read all lines from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Store the last word from each line
line_to_last_word = {}
unique_lines = []
duplicated_last_words = set()

# First pass: find duplicated last words
for line in lines:
    words = line.strip().split()
    if words:
        last_word = words[-1]
        if last_word in line_to_last_word:
            duplicated_last_words.add(last_word)
        else:
            line_to_last_word[last_word] = line

# Second pass: keep only lines with unique last words
for line in lines:
    words = line.strip().split()
    if words and words[-1] not in duplicated_last_words:
        unique_lines.append(line)

# Write the filtered lines to a new file instead of overwriting the original
with open(output_path, 'w') as file:
    file.writelines(unique_lines)

# Count how many lines were removed
lines_removed = len(lines) - len(unique_lines)
print(f"Removed {lines_removed} lines with duplicate last words.")
print(f"New file '{output_path}' contains {len(unique_lines)} lines.")
print(f"Original file '{file_path}' remains unchanged.")
