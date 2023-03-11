import re

names = ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)']
names_with_mr = []

for name in names:
    if re.search(r'\bMr\.\b', name):
        names_with_mr.append(name)
print(names_with_mr)