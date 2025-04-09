from typing import Any

import spacy
import json
from spacy.training.example import Example


nlp = spacy.load("modelo_cor")
'''
print(nlp.pipe_names)

doc = nlp('As ações da Magazine Luiza S.A. são azuis, Franca, Brasil é amarelo, acumularam baixa de 70% ao ano.')

for entidade in doc.ents:
  print("Entidade: ", entidade.text, " Label: ", entidade.label_)


for token in doc:
  print(token.text)


print("Tokens:           ", [token.text for token in doc])
print("Stop word:        ", [token.is_stop for token in doc])
print("Alfanumérico:     ", [token.is_alpha for token in doc])
print("Maiúsculo:        ", [token.is_upper for token in doc])
print("Pontuação:        ", [token.is_punct for token in doc])
print("Número:           ", [token.like_num for token in doc])
print("Sentença Inicial: ", [token.is_sent_start for token in doc])



print("Tokens:  ", [token.text for token in doc])
print("Formato: ", [token.shape_ for token in doc])


#doc = nlp('As ações da Magazine Luiza S.A., Franca, Brasil, acumularam baixa de 70% ao ano.')

for entidade in doc.ents:
  print("Entidade: ", entidade.text, " Label: ", entidade.label_)

'''
# Carregue o modelo pré-treinado em português

#aqui!!!
nlp = spacy.blank('pt')
''''
# Defina as classes de rótulo
LABELS = ["COR"]

ner = nlp.add_pipe("ner")  # Adicionar componente NER
if not ner:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)

for label in LABELS:
    ner.add_label(label)  


#with open('annotations.txt', 'r', encoding='utf-8') as arquivo:
#    conteudo = arquivo.read()


TRAIN_DATA = {
    "classes":["COR"],
        "annotations":[
            ["As cores têm um papel essencial em nossas vidas, trazendo vibração e significado aos nossos ambientes. O azul do céu transmite serenidade e calma, enquanto o verde das folhas nos conecta à natureza e à renovação. O amarelo, associado ao sol, exala alegria e energia. O vermelho, por sua vez, simboliza paixão e força, enquanto o branco remete à pureza e paz. Já o preto é misterioso e elegante, trazendo um contraste marcante. Por fim, cores como o roxo e o laranja oferecem um toque de criatividade e entusiasmo ao cotidiano.\r",
        {"entities":[[105,109,"COR"],[158,163,"COR"],[215,222,"COR"],[329,335,"COR"],[364,369,"COR"],[449,453,"COR"],[458,465,"COR"]]}],
        ["As cores pintam nosso mundo de significados e sensações. O turquesa do oceano nos faz sonhar com destinos tropicais e tranquilidade. O cinza suave de um dia nublado pode trazer reflexão e introspecção. O dourado lembra a realeza e o brilho do pôr do sol. O magenta, com sua energia vibrante, é perfeito para expressar criatividade e ousadia. Já o marrom nos conecta ao solo e à terra, representando estabilidade e conforto. O prateado, com seu brilho metálico, nos remete à modernidade e elegância. E o lilás, doce e suave, fala de delicadeza e espiritualidade. \r",
        {"entities":[[59,67,"COR"],[135,140,"COR"],[204,211,"COR"],[257,264,"COR"],[347,353,"COR"],[426,434,"COR"],[451,459,"COR"],[503,508,"COR"]]}],
        ["Ontem, a cidade de Sinop foi palco de um espetáculo de cores e alegria durante a 10ª Parada Cultural Anual. Pessoas de todas as idades se reuniram vestindo trajes vibrantes que variavam do azul cobalto ao amarelo canário, enquanto carros alegóricos exibiam temas inspirados no verde esmeralda da natureza, no vermelho rubi do amor e no dourado cintilante que simbolizava a luz do sol.\r",
        {"entities":[[189,201,"COR"],[205,220,"COR"],[277,292,"COR"],[309,322,"COR"],[336,354,"COR"]]}],
        ["O evento culminou com uma chuva de confetes prateados e rosas, transformando as ruas em um verdadeiro mar de cores. \"Foi emocionante ver como a diversidade e as cores nos uniram hoje,\" declarou uma participante que vestia um traje com tons de lilás e laranja flamejante.\r",
        {"entities":[[56,61,"COR"],[243,248,"COR"],[251,269,"COR"]]}],
        ["Era uma ves uma menina chamada Júlia. Ela gostava muito do amarelo porque era a cor do sol e deixava ela feliz. Um dia, Júlia pintou um desenho com azul do céu, verde da grama e vermelho das flores. O desenho dela ficou tão bonito que até a professora falou: \"Que lindo, Júlia!\"\r",
        {"entities":[[59,66,"COR"],[148,152,"COR"],[161,166,"COR"],[178,186,"COR"]]}],
        ["No outro dia, Júlia foi no parque e viu um passarinho laranja cantando em cima de uma árvore marrom. Ela falou: \"Oi, passarinho! Você é muito bonito!\" O passarinho cantou mais forte e Júlia sorriu. Antes de ir pra casa, Júlia olhou o céu e viu o roxo do pôr do sol e falou: \"Que legal! Tem um monte de cor hoje!\"",
        {"entities":[[54,61,"COR"],[93,100,"COR"],[246,250,"COR"]]}],["No outro dia, Júlia foi no parque e viu um passarinho laranja cantando em cima de uma árvore marrom. Ela falou: \"Oi, passarinho! Você é muito bonito!\" O passarinho cantou mais forte e Júlia sorriu. Antes de ir pra casa, Júlia olhou o céu e viu o roxo do pôr do sol e falou: \"Que legal! Tem um monte de cor hoje!\"",
        {"entities":[[54,61,"COR"],[93,100,"COR"],[246,250,"COR"]]}],
        ["As cores têm um papel essencial em nossas vidas, trazendo vibração e significado aos nossos ambientes. O azul do céu transmite serenidade e calma, enquanto o verde das folhas nos conecta à natureza e à renovação. O amarelo, associado ao sol, exala alegria e energia. O vermelho, por sua vez, simboliza paixão e força, enquanto o branco remete à pureza e paz. Já o preto é misterioso e elegante, trazendo um contraste marcante. Por fim, cores como o roxo e o laranja oferecem um toque de criatividade e entusiasmo ao cotidiano.\r",
        {"entities":[[105,109,"COR"],[158,163,"COR"],[215,222,"COR"],[329,335,"COR"],[364,369,"COR"],[449,453,"COR"],[458,465,"COR"]]}],
        ["As cores pintam nosso mundo de significados e sensações. O turquesa do oceano nos faz sonhar com destinos tropicais e tranquilidade. O cinza suave de um dia nublado pode trazer reflexão e introspecção. O dourado lembra a realeza e o brilho do pôr do sol. O magenta, com sua energia vibrante, é perfeito para expressar criatividade e ousadia. Já o marrom nos conecta ao solo e à terra, representando estabilidade e conforto. O prateado, com seu brilho metálico, nos remete à modernidade e elegância. E o lilás, doce e suave, fala de delicadeza e espiritualidade. \r",
        {"entities":[[59,67,"COR"],[135,140,"COR"],[204,211,"COR"],[257,264,"COR"],[347,353,"COR"],[426,434,"COR"],[451,459,"COR"],[503,508,"COR"]]}],
        ["Ontem, a cidade de Sinop foi palco de um espetáculo de cores e alegria durante a 10ª Parada Cultural Anual. Pessoas de todas as idades se reuniram vestindo trajes vibrantes que variavam do azul cobalto ao amarelo canário, enquanto carros alegóricos exibiam temas inspirados no verde esmeralda da natureza, no vermelho rubi do amor e no dourado cintilante que simbolizava a luz do sol.\r",
        {"entities":[[189,201,"COR"],[205,220,"COR"],[277,292,"COR"],[309,322,"COR"],[336,354,"COR"]]}],
        ["O evento culminou com uma chuva de confetes prateados e rosas, transformando as ruas em um verdadeiro mar de cores. \"Foi emocionante ver como a diversidade e as cores nos uniram hoje,\" declarou uma participante que vestia um traje com tons de lilás e laranja flamejante.\r",
        {"entities":[[56,61,"COR"],[243,248,"COR"],[251,269,"COR"]]}],
        ["Era uma ves uma menina chamada Júlia. Ela gostava muito do amarelo porque era a cor do sol e deixava ela feliz. Um dia, Júlia pintou um desenho com azul do céu, verde da grama e vermelho das flores. O desenho dela ficou tão bonito que até a professora falou: \"Que lindo, Júlia!\"\r",
        {"entities":[[59,66,"COR"],[148,152,"COR"],[161,166,"COR"],[178,186,"COR"]]}],
        ["No outro dia, Júlia foi no parque e viu um passarinho laranja cantando em cima de uma árvore marrom. Ela falou: \"Oi, passarinho! Você é muito bonito!\" O passarinho cantou mais forte e Júlia sorriu. Antes de ir pra casa, Júlia olhou o céu e viu o roxo do pôr do sol e falou: \"Que legal! Tem um monte de cor hoje!\"",
        {"entities":[[54,61,"COR"],[93,100,"COR"],[246,250,"COR"]]}],
        ["No outro dia, Júlia foi no parque e viu um passarinho laranja cantando em cima de uma árvore marrom. Ela falou: \"Oi, passarinho! Você é muito bonito!\" O passarinho cantou mais forte e Júlia sorriu. Antes de ir pra casa, Júlia olhou o céu e viu o roxo do pôr do sol e falou: \"Que legal! Tem um monte de cor hoje!\"",
        {"entities":[[54,61,"COR"],[93,100,"COR"],[246,250,"COR"]]}]
    ]
    }



# Converta os dados de treinamento em exemplos do spaCy
examples = []
for annotation in TRAIN_DATA.get('annotations', []):
    text = annotation[0]
    entities = annotation[1].get('entities', [])
    example = Example.from_dict(nlp.make_doc(text), {"entities": entities})
    examples.append(example)

# Inicialize e treine o modelo
nlp.begin_training()
losses = {}
for i in range(500):  # Executar iterações de treinamento
    nlp.update(examples, losses=losses)
  #  print(losses)  # Imprimir as perdas de treinamento a cada iteração

# Salve o modelo treinado
nlp.to_disk("modelo_cor")

'''

modelo_treinado = spacy.load("modelo_cor")

doc = modelo_treinado("Numa folha qualquer Eu desenho um sol amarelos E com cinco ou seis retas É fácil fazer um castelo Com o lápis em torno da mão E me dou uma luva E se faço chover Com dois riscos tenho um guarda-chuva Se um pinguinho de tinta Cai num pedacinho azul do papel Num instante imagino Uma linda gaivota a voar no céu Vai voando Contornando a imensa curva, norte, sul Vou com ela viajando Havaí, Pequim ou Istambul Pinto um barco à vela branco, navegando É tanto céu e mar num beijo azul Entre as nuvens vem surgindo Um lindo avião rosa e grená Tudo em volta colorindo Com suas luzes a piscar Basta imaginar e ele está partindo Sereno indo E se a gente quiser Ele vai pousar Numa folha qualquer Eu desenho um navio de partida Com alguns bons amigos Bebendo de bem com a vida De uma América a outra Eu consigo passar num segundo Giro um simples compasso E num círculo eu faço o mundo Um menino caminha E caminhando chega num muro E ali logo em frente A esperar pela gente o futuro está E o futuro é uma astronave Que tentamos pilotar Não tem tempo, nem piedade Nem tem hora de chegar Sem pedir licença, muda a nossa vida E depois convida a rir ou chorar Nessa estrada não nos cabe Conhecer ou ver o que virá O fim dela ninguém sabe Bem ao certo onde vai dar Vamos todos numa linda passarela De uma aquarela Que um dia enfim descolorirá Numa folha qualquer Eu desenho um sol amarelo (que descolorirá) E com cinco ou seis retas É fácil fazer um castelo (que descolorirá) Giro um simples compasso E num círculo eu faço o mundo (que descolorirá)")
docTeste = "Azul-claro, amarelo, verde, brancas, terracota, verde-escuro, beige, vermelhas, azul, dourados, prateados, amarela, roxas, branco, vermelho-escuro, Borgonha, cinza, preta, lilás, douradas, brancos, cinza-claro, dourada, verdes escuras, azul, verde-musgo, lilases, roxos, verde-claro, marrom escuro, vermelho, cinza-escuro, laranja, verde-água, bege, amarelo, preto, amareladas, azul-petróleo, branca,azul escuro,"



textoValidacao = [item.strip() for item in docTeste.split(",")]#organizar a string para comprarar
print(textoValidacao)
# for token in doc:
#   print(token.text)
""

def calculoRecall(tp, fn):
    recall = float(100*(tp / (tp + fn)))
    return recall


def calculoPrecision(tp, fp):
    precision = float(100*(tp / (tp + fp)))
    return precision


def calculoF1(precision, recall):
    f1 = float((2 * ((precision * recall) / (precision + recall))))
    return f1

tp = 0
fn = 0
fp = 0
for entidade in doc.ents:
    print(entidade.text, entidade.label_)
    if entidade.text in textoValidacao:
        tp += 1
    if entidade.text not in textoValidacao:
        fn += 1

fp = len(textoValidacao) - tp
print(tp)
print(fn)
print(fp)

print(f"Recall: {calculoRecall(tp,fn)}")
recall = calculoRecall(tp,fn)
print(f"Precision: {calculoPrecision(tp,fp)}")
precision = calculoPrecision(tp,fp)
print(f"F1: {calculoF1(precision, recall)}")
