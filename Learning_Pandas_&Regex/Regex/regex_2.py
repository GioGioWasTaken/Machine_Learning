import re

emails = ["john.smith@gmail.com", "jane.doe@yahoo.com", "bob.johnson@gmail.com", "sara.lee@hotmail.com"]
gmails = []

for email in emails:
    if re.search(r"\w+@gmail\.com", email): #\w+ makes sure there are alphabetic/neumerical/ _ characters before the @gmail.com portion.
        gmails.append(email)

print(gmails)