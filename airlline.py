import csv
s ='ANC'
de = 'SEA'
a='MN'
#"Airline Checking Code"           
def air_line(s,de,a): 
    path=open('ArrEarlyP.csv')
    cs=csv.reader(path)
    for row in cs:
        if row[3]==s:
            if row[4]==de:
                if row[5]==a:
                    e=row[5]
                    return e
                    break
                
a=air_line(s,de,a)
print (a)