# -*- coding: utf-8 -*-
"""
Base de dados de treino do classificador de sentimentos.

Cada exemplo eh uma tupla (texto, rotulo), onde o rotulo eh:
    "positivo"  -> a frase expressa um sentimento bom
    "negativo"  -> a frase expressa um sentimento ruim

Esta eh a "experiencia" da nossa IA. Ela aprende observando como
as palavras aparecem em frases positivas x negativas.

Quanto mais exemplos, melhor a IA generaliza. Aqui usamos frases no
estilo de avaliacoes de produtos/servicos (reviews).
"""

DADOS_TREINO = [
    # ---------------------- POSITIVOS ----------------------
    ("Adorei o produto, superou minhas expectativas", "positivo"),
    ("Excelente atendimento, muito rapido e educado", "positivo"),
    ("O sabor e maravilhoso, recomendo demais", "positivo"),
    ("Entrega super rapida, chegou antes do prazo", "positivo"),
    ("Produto de otima qualidade, vale cada centavo", "positivo"),
    ("Amei a experiencia, voltarei a comprar com certeza", "positivo"),
    ("Muito bom, funcionou perfeitamente", "positivo"),
    ("Fiquei muito satisfeito com a compra", "positivo"),
    ("Atendimento maravilhoso, equipe simpatica e atenciosa", "positivo"),
    ("A qualidade e incrivel, melhor que eu esperava", "positivo"),
    ("Recomendo, produto otimo e barato", "positivo"),
    ("Perfeito, chegou tudo certinho e bem embalado", "positivo"),
    ("Que delicia, adorei o cheiro e o sabor", "positivo"),
    ("Comprei de novo porque e simplesmente sensacional", "positivo"),
    ("Loja confiavel, produto original e otimo preco", "positivo"),
    ("Melhor compra que fiz esse ano, estou apaixonado", "positivo"),
    ("Servico impecavel, resolveram tudo rapidinho", "positivo"),
    ("Muito satisfeita, produto lindo e de qualidade", "positivo"),
    ("Otimo custo beneficio, valeu muito a pena", "positivo"),
    ("Estou adorando, funciona muito bem e e facil de usar", "positivo"),
    ("Fantastico, superou tudo que eu imaginava", "positivo"),
    ("Recomendo fortemente, produto durável e bonito", "positivo"),
    ("Chegou rapido e o produto e maravilhoso", "positivo"),
    ("Experiencia otima do inicio ao fim, parabens", "positivo"),
    ("Adorei tudo, atendimento nota dez", "positivo"),
    ("Produto top, qualidade excelente e entrega rapida", "positivo"),
    ("Muito feliz com a compra, recomendo para todos", "positivo"),
    ("Sensacional, o melhor que ja usei", "positivo"),
    ("Gostei muito, atendeu perfeitamente o que eu precisava", "positivo"),
    ("Otimo, barato e de boa qualidade", "positivo"),

    # ---------------------- NEGATIVOS ----------------------
    ("Odiei o produto, veio quebrado e sujo", "negativo"),
    ("Pessimo atendimento, demoraram demais para responder", "negativo"),
    ("O sabor e horrivel, joguei tudo fora", "negativo"),
    ("Entrega atrasada, chegou muito depois do prazo", "negativo"),
    ("Produto de pessima qualidade, nao vale o preco", "negativo"),
    ("Detestei, foi uma experiencia terrivel", "negativo"),
    ("Muito ruim, parou de funcionar no primeiro dia", "negativo"),
    ("Fiquei extremamente insatisfeito com a compra", "negativo"),
    ("Atendimento horrivel, funcionarios mal educados", "negativo"),
    ("A qualidade e pessima, quebrou rapidinho", "negativo"),
    ("Nao recomendo, produto caro e ruim", "negativo"),
    ("Veio errado e mal embalado, uma decepcao", "negativo"),
    ("Que nojo, o cheiro e horrivel", "negativo"),
    ("Nunca mais compro aqui, pessima experiencia", "negativo"),
    ("Loja enganosa, produto falsificado e caro", "negativo"),
    ("Pior compra do ano, estou muito arrependido", "negativo"),
    ("Servico terrivel, nao resolveram meu problema", "negativo"),
    ("Muito insatisfeita, produto feio e frágil", "negativo"),
    ("Pessimo custo beneficio, nao valeu a pena", "negativo"),
    ("Estou odiando, nao funciona e e dificil de usar", "negativo"),
    ("Horrivel, muito pior do que eu imaginava", "negativo"),
    ("Nao recomendo, produto quebradico e feio", "negativo"),
    ("Demorou uma eternidade e ainda veio com defeito", "negativo"),
    ("Experiencia pessima do inicio ao fim, uma vergonha", "negativo"),
    ("Detestei tudo, atendimento nota zero", "negativo"),
    ("Produto ruim, qualidade pessima e entrega atrasada", "negativo"),
    ("Muito triste com a compra, uma grande decepcao", "negativo"),
    ("Terrivel, o pior que ja usei", "negativo"),
    ("Nao gostei, nao atendeu o que eu precisava", "negativo"),
    ("Pessimo, caro e de baixa qualidade", "negativo"),
]

# Frases separadas para TESTE (a IA nunca viu essas na hora de aprender).
# Usadas para medir a acuracia (quantas ela acerta).
DADOS_TESTE = [
    ("Adorei, produto excelente e entrega rapida", "positivo"),
    ("Muito satisfeito, recomendo demais", "positivo"),
    ("Que maravilha, superou minhas expectativas", "positivo"),
    ("Otimo atendimento e otima qualidade", "positivo"),
    ("Amei tudo, voltarei a comprar", "positivo"),
    ("Odiei, produto horrivel e caro", "negativo"),
    ("Pessimo, chegou quebrado e atrasado", "negativo"),
    ("Muito ruim, nao recomendo", "negativo"),
    ("Atendimento terrivel, uma decepcao", "negativo"),
    ("Detestei, pior compra do ano", "negativo"),
]
