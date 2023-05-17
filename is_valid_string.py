def is_valid_string(s):
    # Count the frequency of each character
    char_counts = {}
    for char in s:
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] = 1

    # Check if all frequencies are the same
    frequencies = list(char_counts.values())
    if len(set(frequencies)) == 1:
        return "YES"

    # Check if removing one character can make all frequencies the same
    for char in char_counts:
        char_counts[char] -= 1
        updated_frequencies = list(char_counts.values())
        if len(set(updated_frequencies)) == 1:
            return "YES"
        char_counts[char] += 1

    return "NO"

#Test case 1
s = "aabbcc"
print(is_valid_string(s))
#Output = YES

#Test case 2
s= "aabcc"
print(is_valid_string(s))
#oUTPUT = NO

#Test Case 3 
s = "abcdcc"
print(is_valid_string(s))
#Output No