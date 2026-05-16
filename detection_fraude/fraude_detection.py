import math
import matplotlib.pyplot as plt


Montant = [50, 100, 170, 450, 600, 2000, 6000, 10000]
Heure = [14 ,16 ,2 ,1 ,10, 0,7 ,3]
Nb_transaction = [1, 3 , 5 ,7 ,9, 13,16,20]
Pays_risque = [0 , 1 , 1 ,0 , 1, 1 , 0 , 0 ]

# Longueur ny liste rehetra ireo 
n = len(Montant)
print("Longueur montant = ", n)

# ato dia mijery fotsiny ireo moyenne mba afatarana ny seuil normale ,ka izayb mihoatra ny moyenne dia eritreretina fa fraude 
def mean(x) :
    return sum(x)/len(x)
def variance (x) :
    n = mean(x)
    return sum((xi-n)**2 for xi in x)/len(x)
def std(x):
    return math.sqrt(variance(x))
def corr(x,y):
    mx = mean(x)
    my = mean (y)

    num = sum((x[i]-mx)*(y[i]- my) for i in range (len(x)))
    den = math.sqrt(
       ( sum((x[i]-mx)**2) for i in range(len(x))) *
       ( sum((y[i]-my)**2 for i in range(len(y))))
        )
    return num/den
    
    
moyenne = mean(Montant)
ecart_type = std (Montant)
seuil =  moyenne + 2 * ecart_type
moyenne_transaction = mean(Nb_transaction)

print("amount mean = " , moyenne)
print("ammount std = ",ecart_type )
print ("seuil de transaction = ",seuil)
print("Correlation = ",corr(Montant,Nb_transaction))
#===================Suspects=========================
suspects =[]
for i in range (n) :
    if Montant[i]> seuil:
        suspects.append(i)
if len(suspects) == 0 :
    print ("Empty DataFrame")
    print("Column:[Montant ,Heure ,nb_transactions , Pays_risque , Fraude]")
    print("Index : []")
else:
 print(suspects)

#================resultat detection de fraude ========================
print("\n Montant Heure Nb_transaction Pays_risque Resultat ")
for i in range(n):
    if Montant[i] > seuil or Nb_transaction[i] > 5 or Pays_risque[i] == 1:
        res = "FRAUDE SUSPECTE"
    else:
        res = "NORMALE"
    print(f"{Montant[i]} {Heure[i]} {Nb_transaction} {Pays_risque[i]} {res}")


#=================Probabilite fraude=====================
P_F = sum (Pays_risque)/n
print("probabilité fraude : ",P_F)

#=================Graphs==================================
plt.hist(Montant)
plt.title("Distrubition des montants")
plt.xlabel("Montant")
plt.ylabel("Frenquence")
plt.legend()
plt.show()

#========Graphs de correlation entre Montant et Nb_transaction====
plt.scatter( Nb_transaction,Montant) # manamparitaka
plt.title("Correlation entre montant et nombre de transaction")
plt.ylabel("Montant")
plt.xlabel("Nb_transaction")
plt.show()