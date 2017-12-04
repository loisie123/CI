
# voor elke generatie:
#     voor elke populatie:
#
#         voor elk individu:
#             start de race:
#
#
#             sla de fitnesscore op
#
from ea2 import *
from pathlib import Path
import os

#populations = makepopulation(1, parents_file ='/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/ouwedata.pt')
#torch.save(populations[1], 'species_1.pt')


populations = torch.load('species_1.pt')
print(len(populations))


net = populations[0]

#Verwijder netwerk uit de lijst
populations.pop(0)
torch.save(populations, 'species_1.pt')


print(len(populations))

my_file = Path('/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/children_trying.pt')
if my_file.is_file():
    print("bestand bestaat dus wordt toegevoegd aan bestaande lijst")
    old_net = torch.load('children_trying.pt')
    new_net = (net,230)
    old_net.append(new_net)
    torch.save(old_net, 'children_trying.pt')
else:
    print("bestand bestaat niet")
    new_net = [(net,230)]
    torch.save(new_net, 'children_trying.pt')




if len(populations) == 0:
   print ("de ouders gaan seksen" )
   print("kinderen worden nu in species_1 gezet")
   children = torch.load('children_trying.pt')
   torch.save(children, 'species_1.pt')
   # make children en sla deze op 
else:
    print("tijd voor volgende netwerk dan ")
    print("ik ga nu restarten")


#
#
# te doen:
# de auto gaat niet rijden als de gear niet veranderd
# er moet dus een functie geschreven worden om de gear te veranderen.


# er moet een manier gevonden woden dat de mydriver het programma stopt.
# zodra deze stopt neemt deze file het over en gaat het nog een keer uitvoeren.
# kan ervoor zorgen dat we we files kunnen opslaan. en deze weer kunnen openen.
