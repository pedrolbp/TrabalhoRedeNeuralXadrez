# Autor : Pedro Haro
# motivo : criacao da IA de xadrez

# tem que pagar pra conseguir usar o console, ent eu instalo a
# biblioteca diretamento do interpretador (Não sei se isso é legal)
import sys
!{sys.executable} -m pip install chess

# outros imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from keras import layers, models, regularizers
import numpy as np
import pandas as pd
import chess
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Add, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

import os
from google.colab import drive

# tendo acesso ao meu drive
drive.mount('/content/drive')

# definindo o caminho do dataset
data_path = '/content/drive/MyDrive/chessDataSet/jogos_unificados.csv'

#                         #
# Processamento dos Dados #
#                         #



# Converto Strings formatadas em FEN para tensores
# tensores de 8 * 8 * 12
def fen_para_tensor(fen, piece_map=None):
    if piece_map is None:
       piece_map = {
                    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, # pecas Brancas
                    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # pecas Negras
       }
    # crio a array, 8 * 8 é o numero de quadrados e o 12 sao as pecas
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # Pego apenas o estado do tabuleiro
    rows = fen.split()[0].split('/')
    for row_idx, row in enumerate(rows):
        # coluna do tensor
        col_idx = 0
        for char in row:
            if char.isdigit():
                # se o char for um valor numerico, significa que ele não é uma peca
                # entao esse valor ao numero de colunas do tensor
                col_idx += int(char)
            else:
                # caso contrario, eu adciono ao tensor
                tensor[piece_map[char], row_idx, col_idx] = 1.0
                col_idx += 1
    return tensor

# UCI é uma string que representa movimentos de xadrez, a IA não entende STRINGS
# então eu converto elas para um valor numerico
def uci_para_index(uci_move):
    # Se o movimento tiver 5 chars, ele é uma promocao
    if len(uci_move) == 5:
        # pego apenas a peca que ele se tornou
        uci_move = uci_move[:4]
    # Valores dos quadrados
    start_square = chess.SQUARE_NAMES.index(uci_move[:2])
    end_square = chess.SQUARE_NAMES.index(uci_move[2:])

    # Essa formula não é especial, o meu unico objetivo é apenas fazer a conver
    # sao para um valor numerico que seja unico e fassa possivel a generalizacao
    return start_square * 64 + end_square


# Essa funcao serve para carregar uma pequena parte do .csv, e processar essa parte
# para a IA usar, isso é necessário pq eu não vou pagar o google collab para
# conseguir mais RAM, ent essa funcao basicamente me ajuda a optimizar tudo
def load_and_preprocess_subset(filepath, skip_rows, nrows, piece_map=None):
    # Pocesso partes do .csv, e crio uma coluna chamada outcome_numeric, essa
    # coluna dita que as strings que representam os possiveis resultados tem
    # valores numericos especificos
    chunk = pd.read_csv(filepath, skiprows=range(1, skip_rows + 1), nrows=nrows)
    chunk['outcome_numeric'] = chunk['outcome'].map({"1-0": 1, "0-1": -1, "1/2-1/2": 0})

    # tensores
    board_tensors = []
    move_indices = []
    outcomes = []

    # Debug, uso esse valor para setar os pesos dos possiveis resultados
    print(chunk['outcome'].value_counts())

    # seto os tensores a os valores processados
    for _, row in chunk.iterrows():
        board_tensors.append(fen_para_tensor(row['board_state'], piece_map))
        move_indices.append(uci_para_index(row['move']))
        outcomes.append(row['outcome_numeric'])

    board_tensors = np.array(board_tensors)
    move_indices = np.array(move_indices)
    outcomes = np.array(outcomes)

    return board_tensors, move_indices, outcomes

#                   #
# Criacao do Modelo #
#                   #


from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add

def create_model(input_shape=(12, 8, 8)):
    input_layer = Input(shape=input_shape)

    # bloco convulacional inicial
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # bloco secundario
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # bloco ternario
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # global pooling e camada densa
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)

    # a IA tem 4096 possiveis jogadas
    output_layer = Dense(4096, activation='softmax')(x)

    # compilacao da ia
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
    return model



# geracao da tabela e criacao
model = create_model()
model.summary()



# geracao da tabela e criacao
model = create_model()
model.summary()

#        #
# BackUp #
#        #



lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)

# definicao do do backup
checkpoint_path = "/content/drive/MyDrive/chess_model_checkpoint.keras"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=False,
    verbose=1
)

#                   #
# Treinamento da IA #
#                   #

from sklearn.utils.class_weight import compute_class_weight

# existe uma discrepancia na divisao de resultados
classes = ["1-0", "0-1", "1/2-1/2"]
# Essa é a divisao [NumDeVitorias, NumeDeDerrotas, NumDeEmpates]
counts = [42061, 37539, 20400] # Estes valores foram pegos do dataset analisando
# a ultima coluna
# essa é a soma
total = sum(counts)

# os pesos sao usados para balancear o valor da importancia que a IA dara
# a cada classe
class_weights = {i: total / (len(classes) * count) for i, count in enumerate(counts)}

# Melhort coisa que eu fiz foi aprender a como fazer backups de AI
# agora eu posso parar o treinamento a qualquer momento e fazer modificacoes
checkpoint_path = "/content/drive/MyDrive/chess_model_checkpoint.keras"

if os.path.exists(checkpoint_path):
    print("Carregando backup")
    model = tf.keras.models.load_model(checkpoint_path)
    print("Backup carregado")
else:
    raise FileNotFoundError("BackUp n encontrado")

# Toda vez que eu pauso, eu tenho que lembrar de modificar essas variaveis
resume_epoch = 1 # época na hora da pausa
start_row = 0  # batche e que estava a ser executado

# Paramentros de treinamento
batch_size = 250000 # quantidade de linhas que a IA treina em cada execucao
total_rows = 32658400  # Numero total de linhas do .csv
validation_split = 0.1
epochs = 2

# Loop de treinamento, determinado pelo numero de epocas
for epoch in range(resume_epoch, epochs):
    # debug, para quando eu pausar eu saber o que substituir nas variaveis la
    # de cima
    # treino com base no batch
    for current_row in range(start_row, total_rows, batch_size):
        # coluna limite, usada no debug
        end_row = min(current_row + batch_size, total_rows)
        print(f"Carregando linhas {current_row} ate {end_row}...")

        # carrego e processo o batch
        board_tensors, move_indices, game_outcomes = load_and_preprocess_subset(
        data_path, skip_rows=current_row, nrows=batch_size
        )
        print(f"subset carregado com {len(board_tensors)} linhas.")

        # Treinamento e validacao
        train_size = int(len(board_tensors) * (1 - validation_split))
        val_inputs, val_labels = board_tensors[train_size:], move_indices[train_size:]
        train_inputs, train_labels = board_tensors[:train_size], move_indices[:train_size]

        print("Preprocessamento completo.")
        print(f"Treinando nas linhas {current_row} ate {end_row}...")

        # Treino o modelo no batch que eu dividi
        model.fit(
            # variaveis de treinamento e validacao
            train_inputs, train_labels,
            validation_data=(val_inputs, val_labels),
            # "Passos" dentro do batch principal, esse é um batch do batch
            batch_size=1024,
            # Numero de Bathcs dos Batchs
            epochs=1,
            # Pesos das clases de resultados
            class_weight=class_weights,
            # Coisas que sao executadas depois do treinamento, nesse caso
            # é o backup, e eu mostro dados adicionais
            callbacks=[checkpoint_callback, lr_callback]
        )

        # Limpo da memoria o batch dps de cada treinamento
        del board_tensors, move_indices, game_outcomes
        del train_inputs, train_labels, val_inputs, val_labels

    # linha inicial reseta
    start_row = 0
    print("Deu certo amigao")

#               #
# Salvando a IA #
#               #

import tensorflow as tf

keras_model_path = "/content/drive/MyDrive/chess_model_checkpoint.keras"

model = tf.keras.models.load_model(keras_model_path)

h5_model_path = "/content/drive/MyDrive/chess_model_checkpoint.h5"
model.save(h5_model_path)

print(f"Model converted and saved as: {h5_model_path}")

import tensorflow as tf

#             #
# Convertendo #
#             #

h5_model_path = "/content/drive/MyDrive/chess_model_checkpoint.h5"
model = tf.keras.models.load_model(h5_model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_path = "/content/drive/MyDrive/chess_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_model_path}")
