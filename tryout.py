import torch

new_pop = []
pop = torch.load('/home/student/Desktop/CI/species_extra_2.pt')
print (len(pop))

for i in pop:
    new_pop.append(i)

pop1 = torch.load('/home/student/Desktop/CI/species_extra_1.pt')
print (len(pop1))

for i in pop1:
    new_pop.append(i)

print(len(new_pop))

torch.save(new_pop, 'complete_2_family.pt')
