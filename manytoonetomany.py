# many to one data structure 

monty = {
     ('parrot','spam','cheese_shop'): 'sketch', 
    ('Cleese', 'Gilliam', 'Palin') : 'actors',
}


working_monty = {}
for k, v in monty.items():
    for key in k:
        working_monty[key] = v

    print(working_monty)



