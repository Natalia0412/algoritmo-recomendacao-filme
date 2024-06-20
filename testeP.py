import pandas as pd
import numpy as np

# Carregando os dados
caminho = "C:\\Users\\natal\\OneDrive\\Área de Trabalho\\Estudo Python\\SistemaRecomendação\\ml-latest-small"
caminho_arquivo = caminho + "\\movies.csv"
filmes = pd.read_csv(caminho_arquivo)
filmes.columns = ["filmeId", "titulo", "genero"]
filmes = filmes.set_index("filmeId")

caminho_arquivo = caminho + "\\ratings.csv"
notas = pd.read_csv(caminho_arquivo)
notas.columns = ["usuarioId", "filmeId", "nota", "momento"]

# Calculando o total de votos por filme e a média das notas por filme
total_de_votos = notas["filmeId"].value_counts()
filmes['total_de_votos'] = total_de_votos

nota_medias = notas.groupby("filmeId").mean()["nota"]
filmes["nota_media"] = nota_medias

# Filtrando filmes com mais de cinquenta votos e mostrando os 10 melhores por nota média
filmes_com_mais_de_cinquenta_votos = filmes.query("total_de_votos > 50")
print(filmes_com_mais_de_cinquenta_votos.sort_values("nota_media", ascending=False).head(10))

# Função para calcular a distância entre dois vetores
def distancia_de_vetores(a, b):
    return np.linalg.norm(a - b)

# Função para obter as notas de um usuário
def notas_do_usuario(usuario):
    notas_do_usuario = notas.query("usuarioId == %d" % usuario)
    notas_do_usuario = notas_do_usuario[["filmeId", "nota"]].set_index(["filmeId"])
    return notas_do_usuario

# Função para calcular a distância entre dois usuários
def distancia_de_usuarios(usuarioId1, usuarioId2):
    notas1 = notas_do_usuario(usuarioId1)
    notas2 = notas_do_usuario(usuarioId2)
    diferencas = notas1.join(notas2, lsuffix="_esquerda", rsuffix="_direita").dropna()
    distancia = distancia_de_vetores(diferencas['nota_esquerda'], diferencas['nota_direita'])
    return [usuarioId1, usuarioId2, distancia]

# Função para calcular as distâncias de todos os usuários
def distancia_de_todos(voce_id):
    distancias = []
    for usuario_Id in notas["usuarioId"].unique():
        informacoes = distancia_de_usuarios(voce_id, usuario_Id)
        distancias.append(informacoes)
    return distancias

# Função KNN para encontrar os usuários mais próximos
def knn(voce_id, k_mais_proximos=10):
    distancias = distancia_de_todos(voce_id)
    distancias_df = pd.DataFrame(distancias, columns=['usuarioId1', 'usuarioId2', 'distancia'])
    distancias_df = distancias_df.sort_values('distancia')
    distancias_df = distancias_df.set_index('usuarioId2').drop(voce_id, errors='ignore')
    return distancias_df.head(k_mais_proximos)

# Função para recomendar filmes para um usuário
def sugere_para(voce, k_mais_proximos=10):
    notas_de_voce = notas_do_usuario(voce)
    filmes_que_voce_ja_viu = notas_de_voce.index

    similares = knn(voce, k_mais_proximos=k_mais_proximos)
    usuarios_similares = similares.index
    notas_dos_similares = notas.set_index("usuarioId").loc[usuarios_similares]
    recomendacoes = notas_dos_similares.groupby("filmeId").mean()[["nota"]]
    aparicoes = notas_dos_similares.groupby("filmeId").count()[['nota']]

    filtro_minimo = k_mais_proximos / 2
    recomendacoes = recomendacoes.join(aparicoes, lsuffix="_media_dos_usuarios", rsuffix="_aparicoes_nos_usuarios")
    recomendacoes = recomendacoes.query("nota_aparicoes_nos_usuarios >= %.2f" % filtro_minimo)
    recomendacoes = recomendacoes.sort_values("nota_media_dos_usuarios", ascending=False)
    recomendacoes = recomendacoes.drop(filmes_que_voce_ja_viu, errors='ignore')
    return recomendacoes.join(filmes)

# Exemplo de uso
print(sugere_para(1, k_mais_proximos=10))

# Mostrando os 10 filmes mais votados
print(filmes.sort_values("total_de_votos", ascending=False).head(10))
