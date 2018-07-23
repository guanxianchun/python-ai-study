import random, string
special_alpha = '#!@$&?_-'
s = string.ascii_letters
list = random.Random().sample([x for x in s], random.Random().randint(8, 20))
list.append(random.choice(special_alpha))
list.extend(random.Random().sample([str(i) for i in range(0, 10)], 2))
random.Random().shuffle(list)
print list
print "".join(list)
# print "".join(.append(random.choice(special_alpha)))
