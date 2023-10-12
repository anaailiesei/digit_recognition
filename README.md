# digit_recognition
Classifier model in MATLAB for distinguishing single digit numbers.

*load_dataset.m*

    1. Se folosetse load pentru x si y
    2. Rezulta doua structuri si se extrag matricea si vectorul corespunzatoare

*split_dataset*

    1. Lipim vectorul y la inceputul matricei X ca sa ne fie mai usor dupa. Rezulta o matrice B
    2. Generam o permutare random pentru linii
    3. Se amesteca liniile matricei B
    4. Se depart liniile din B pentru setul de date train si setul de date test
    5. Apoi se dezlipesc X si y si se realizeaza X train, y train, X test, y test

*initialize_weights.m*

    1. Se calculeaza epsilon cu formula din material
    2. Se creeaza o matrice cu elemente random (de la 0 la 1) de dimensiunile potrivite
    3. Se inmulteste cu 2 si se scade 1 pentru a obtine in matrice vaalori de la -1 la 1
    4. Se inmulteste cu epsilon ca sa se obtina valori de la -epsilon la epsilon

*cost_function.m*

    1. Am folosit formatul long pt o acuratete mai mare
    2. Se realizeaza matricele theta 1 si theta 2 cu reshape din vectorul params
    3. Se realizeaza forward propagation
    4. Se adauga elemenetele de bias lui X si rezulta a1
    5. Se determina outputul din hidden layer a2
    6. se determina outputul din output layer a3
    7. Se realizeaza backpropagation
    8. Se expandeaza y
    9. Se aplica formulele pentru a afla delta 2
    10. Se calculeaza sigam derivat de z2
    11. Se elimina primul rand din primul operand al inmultirii pentru del2 si se inmulteste cu sigma derivat pentru del2 (inmultire element cu element)
    12. Se calculeaza delta 1
    13. Se determina gradientii
    14. Se calculeaza costul cu formulele din enunt (vectorizat)
    15. Se calculeaza cele 3 sume din formula separat si apoi se folosesc in formula finala pentru cost

*predict_classes.m*

    1. Primii pasi sunt la fel ca la cost_function. Se realizeaza forward propagation
    . Se realizeaza matricele theta 1 si theta 2 cu reshape din vectorul params
    3. Se realizeaza forward propagation
    4. Se adauga elemenetele de bias lui X si rezulta a1
    5. Se determina outputul din hidden layer a2
    6. Se determina outputul din output layer a3
    7. Classes va fi un vector cu indicele coloanei cu probabilitatea cea mai mare din output de pe fiecare rand (output are initial pe fiecare coloana probabilitatile pentru o clasa, dar am transpus matricea output)
