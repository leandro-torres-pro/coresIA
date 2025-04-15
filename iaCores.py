from typing import Any
import spacy
import json
from spacy.training.example import Example

'''
nlp = spacy.load("modelo_cor")

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


# Carregue o modelo pré-treinado em português

#aqui!!!
nlp = spacy.blank('pt')

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

doc = modelo_treinado("O céu da manhã vestia um azul-claro suave, tão tranquilo como o mar calmo em uma praia isolada. As flores de uma planta no jardim, com seus pétalos amarelos vibrantes, pareciam pequenas estrelas no meio do verde intenso da grama. A casa, de paredes brancas e telhado terracota, refletia a luz do sol de maneira acolhedora, como se convidasse todos a entrar. No canto da sala, uma poltrona de veludo verde-escuro oferecia um contraste acolhedor contra o tapete beige que cobria o piso de madeira escura. Na cozinha, as maçãs vermelhas brilhavam em uma tigela de cerâmica azul, suas superfícies polidas como jóias. O relógio de parede, com números dourados e ponteiros prateados, marcava as horas em um ritmo calmo, enquanto a parede amarela refletia a luz suave que entrava pela janela. Um vaso de flores roxas, com suas pétalas exuberantes, ficava sobre a mesa de jantar, criando um ponto de cor contrastante com o branco da toalha de linho. A bicicleta encostada no portão era pintada de um vermelho-escuro, quase borgonha, contrastando com o cinza do asfalto da rua. Um conjunto de livros de capa preta estava empilhado em uma prateleira, ao lado de um caderno de capa lilás, que se destacava pela suavidade do tom. A luz do fim da tarde projetava sombras longas e douradas sobre o chão de ladrilhos brancos. No corredor, uma lâmpada de vidro opaco lançava um brilho suave sobre a parede cinza-claro. O espelho de moldura dourada na parede refletia a luz, criando um jogo de cores e formas. Ao lado, uma pequena planta de folhas verdes escuras, quase negras, trazia um pouco da natureza para dentro de casa. No fundo, uma porta de madeira pintada de azul-marinho contrastava com as paredes de tom neutro. No jardim, o banco de ferro pintado de verde-musgo parecia quase fundir-se com o ambiente ao seu redor, enquanto as flores de lavanda em tons lilases e roxos exalavam uma fragrância suave no ar. As pedras brancas do caminho serpenteavam entre as plantas, criando uma sensação de paz e tranquilidade. A grama, em um tom verde-claro, cobria o solo com uma maciez que contrastava com o tom forte da madeira escura do banco. No hall de entrada, uma luminária de cristal refletia a luz, criando uma dança de brilhos e sombras sobre o piso de mármore branco. A porta de entrada, de um marrom escuro, parecia robusta e sólida, acolhendo todos que chegavam. O tapete persa, com seus tons de vermelho e azul, acolhia os pés que pisavam no chão frio, aquecendo o ambiente. A sala de estar, com suas cortinas de veludo cinza-escuro, era complementada por almofadas de cores vivas: laranja, vermelho, azul e até um verde-água. O sofá, de um bege suave, contrastava com os quadros coloridos na parede. Cada obra de arte, com seus tons vibrantes de vermelho, azul, amarelo e preto, trazia uma energia única para o ambiente. O fogo da lareira, em chamas laranja e amareladas, aquecia o ambiente, criando um contraste com o azul profundo da noite que se aproximava. No jardim de inverno, as folhas das plantas tropicais se destacavam em diferentes tons de verde. O tapete macio sob os pés era de um azul-petróleo, dando um toque de sofisticação ao ambiente. O vaso de cerâmica, pintado à mão em tons de terracota, trazia um calor acolhedor, enquanto os refletores de luz branca iluminavam suavemente a cena, criando sombras delicadas no piso. Cada detalhe, desde os móveis de madeira escura até os objetos de decoração coloridos, formava um quadro harmônico e equilibrado. O contraste entre as cores vibrantes e as neutras trazia uma sensação de acolhimento e tranquilidade. No fundo, o céu azul escuro anunciava o fim do dia, mas dentro de casa, a luz suave das lâmpadas criava um ambiente acolhedor, como se o tempo tivesse parado ali.")
docTeste = "Azul-claro, amarelo, verde, brancas, terracota, verde-escuro, beige, vermelhas, azul, dourados, prateados, amarela, roxas, branco, vermelho-escuro, Borgonha, cinza, preta, lilás, douradas, brancos, cinza-claro, dourada, verdes escuras, azul, verde-musgo, lilases, roxos, verde-claro, marrom escuro, vermelho, cinza-escuro, laranja, verde-água, bege, amarelo, preto, amareladas, azul-petróleo, branca,azul escuro,"



textoValidacao = [item.strip() for item in docTeste.split(",")]#organizar a string para comprarar
print(textoValidacao)
print(len(textoValidacao))
# for token in doc:
#   print(token.text)


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
