import json

# Reading the content of the file with UTF-8 encoding
with open('C:\\Users\\Abhishek Poswal\\Desktop\\research\\NLP-Research\\datasets\\json_datasets\\sentence.txt', 'r', encoding='utf-8') as file:
    content = file.readlines()

# Stripping newline characters and converting to JSON format
json_content = []
for line in content:
    # Using eval to parse the string as dictionary
    json_content.append(eval(line.strip()))

# Writing the JSON formatted content to a new file
with open('C:\\Users\\Abhishek Poswal\\Desktop\\research\\NLP-Research\\datasets\\json_datasets\\gpt_35_turbo_prompt_intensity.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_content, json_file, indent=4)

print("Content has been successfully converted to JSON and saved in 'sentences.json'")
