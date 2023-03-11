import re
text=input("Enter the text you'd like cleaned: ")
clean_text=re.sub('[^a-zA-Z0-9]','',text).lower() # when put in [] ^ becomes a 'not' statement.
print(clean_text)