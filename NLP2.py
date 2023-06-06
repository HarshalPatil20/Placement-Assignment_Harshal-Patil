import csv
import PyPDF2
from collections import defaultdict

# Open the PDF file
pdf_path = 'your_pdf_file.pdf'
pdf_file = open(pdf_path, 'rb')

# Read and extract text from the PDF
pdf_reader = PyPDF2.PdfReader(pdf_file)
num_pages = len(pdf_reader.pages)

text = ''
for page_number in range(num_pages):
    page = pdf_reader.pages[page_number]
    text += page.extract_text()

# Text preprocessing
text = text.lower()
text = text.replace('\n', ' ')

# Count word occurrences
word_count = defaultdict(int)
words = text.split()
for word in words:
    word_count[word] += 1

# Find the most repeated word
most_repeated_word = max(word_count, key=word_count.get)
repeated_count = word_count[most_repeated_word]

# Store results in a CSV file
csv_file = open('pdf_text.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Text'])
csv_writer.writerow([text])
csv_file.close()

# Print the most repeated word
print("Most Repeated Word:", most_repeated_word)
print("Count:", repeated_count)
