from sklearn import tree
from numpy.random import seed
from numpy.random import randint
from numpy.random import rand
from numpy.random import normal
from sklearn.naive_bayes import GaussianNB



dados = []
resposta = []
def carregarDados(iteracoes):
    '''
    Testes de coagulação (Categórico: 0, 1)
    Eletrólitos (discreto)
    Hematócrito (continuo, porcentagem)
    TGO (discreto)
    TGP (discreto)
    Contagem de plaquetas (discreto, em milhares)
    Testes sorológicos (categorico: 0, 1)
    Raio X do tórax (categorico: 0, 1)
    febre (continuo)
    dor de cabeça (categórico)
    dor nos olhos(categórico)
    mancha na pele (categorico)
    extremo cansaço (categorico)
    moleza (categorico)
    dor ossos (categorico)
    nauseas (categorico)
    tontura (categorico)
    perda apetite (categorico)
    '''

    for i in range(iteracoes):

        coagulacao = randint(0, 2, 1)[0]
        eletrolitos = int(normal(120, 10))
        hematocrito = rand()
        tgo = int(normal(131, 10))
        tgp = int(normal(99, 10))
        plaquetas = int(normal(150, 10))
        sorologico = randint(0, 2, 1)[0]
        raio_x = randint(0, 2, 1)[0]
        febre = int(normal(38, 2))
        dor_cabeca = randint(0, 2, 1)[0]
        dor_olhos = randint(0, 2, 1)[0]
        mancha_pele = randint(0, 2, 1)[0]
        cansaco = randint(0, 2, 1)[0]
        moleza = randint(0, 2, 1)[0]
        dor_ossos = randint(0, 2, 1)[0]
        nauseas = randint(0, 2, 1)[0]
        tontura = randint(0, 2, 1)[0]
        perda_apetite = randint(0, 2, 1)[0]

        linha = [coagulacao, eletrolitos, hematocrito, tgo, tgp, plaquetas, sorologico, raio_x, febre, dor_cabeca, dor_olhos,
                      mancha_pele, cansaco, moleza, dor_ossos, nauseas, tontura, perda_apetite]
        #print(linha)
        resposta.append(determinaDoente(linha))
        #print(resposta[-1])
        dados.append(linha)


def determinaDoente(linha):
    #D = (TC + E + 3 * (H=1) + 2 * TGO) + 2 * TGP + 3 * C + TS + 1, 5 * Rx) + (
    #            F + DC + DO + M + EC + MC + DA + NV + T + P)
    D_critico = 17
    soma = 0
    for num, l in enumerate(linha):
        if (num == 1):
            if(l<=120):
                soma = soma + 1
        elif (num == 2):
            if (l > 0.55):
                soma = soma + 3
        elif (num == 3):
            if (l > 131):
                soma = soma + 2
        elif (num == 4):
            if (l > 99):
                soma = soma + 2
        elif (num == 5):
            if (l < 150):
                soma = soma + 3
        elif (num == 7):
                soma = soma + (1.5*l)
        elif (num == 8):
            if (l > 39):
                soma = soma + 1
        else:
            soma = soma + 1

    if soma > D_critico:
        return 1
    else:
        return 0

seed(100)
carregarDados(1000)

#print(sum(resposta))

X = dados
y = resposta

#Arvore de decisão
arvore = tree.DecisionTreeClassifier()
arvore.fit(X,y)

#Naive Bayes
bayes = GaussianNB()
bayes.fit(X,y)

seed(1)
dados = []
resposta = []
carregarDados(1)
paciente = dados

print("Paciente: ")
print(paciente)
print("Resultado Arvore: ")
print(arvore.predict(paciente))

print("Resultado Bayes: ")
print(bayes.predict(paciente))

seed(200)
dados = []
resposta = []
carregarDados(100)
treino = dados
resp_treino = resposta


print("Avaliação Arvore: ")
print(arvore.score(treino, resp_treino))

print("Avaliação Bayes: ")
print(bayes.score(treino, resp_treino))