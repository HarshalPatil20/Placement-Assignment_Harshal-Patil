def highest_frequency_length(string):
    words = string.split()


    words_count = {}
    for word in words:
        if word in words_count:
            words_count[word]+=1
        else:
            words_count[word] = 1 



    max_frequency = max(words_count.values())

    max_length = max(len(word) for word, count in words_count.items() if count == max_frequency)


    return max_length

#test case 1
string = "write write write all the number from from from 1 to 100"
print(highest_frequency_length(string))
#Output ---> 5 

#test case 2
#string = "banana apple banana apple mango mango mango mango apple apple apple banana"
#print(highest_frequency_length(string))
#output ----> 5