# standard libraries

# 3rd party libraries

# my libraries


# read file
f = open("WebScrapper.txt", "r")
text = f.readlines()
f.close()

# filter file
filtered_text = []
for line in text:
    if line.find('data=') >= 0:
        line = line.replace('\n','')
        line = line.replace('"','')
        line = line.replace(' ','')
        line = line.replace('second-data=','')
        line = line.replace('data=','')
        line = line.replace('></pk-list-stat-item>','')
        
        filtered_text.append(float(line))

# write result
f = open("WebScrapper.txt", "w")
for line in filtered_text:
    print(line, file = f)
f.close()