# Autor : Pedro haro
# Motivo : Conversão de arquivos .pgn para .csv, que então serão convertidos para tensors
import os
import pandas as pd
import chess.pgn

def pgnParaCsv(arquivoPgn, IDjogo):
    data = []
    game_id = IDjogo 

    with open(arquivoPgn) as f:
        game = chess.pgn.read_game(f)
        while game:
            # board é o tabuleiro de chadrez em formato FEN
            board = game.board()
            # recolho APENAS as informacoes necessárias no .csv
            move_number = 0
            for move in game.mainline_moves():
                # move é uma stack
                board.push(move)
                move_number += 1
                data.append({
                    'game_id': game_id,
                    'move_number': move_number,
                    'board_state': board.fen(),
                    'move': move.uci(),
                    'outcome': game.headers.get("Result")
                })
                # numero dejogos é atualizado
            game_id += 1
            game = chess.pgn.read_game(f)

    # retorno o arquivo .csv
    return pd.DataFrame(data)

# converto todos os arquivos .pgn de um diretorio passados com o PATH completo para .csv
def convert_folder_to_csv(input_folder, output_csv):
    game_id_offset = 0
    first_file = True
    for pgn_file in os.listdir(input_folder):
        if pgn_file.endswith(".pgn"):
            file_path = os.path.join(input_folder, pgn_file)
            print(f"Processado {file_path}")
            df = pgnParaCsv(file_path, game_id_offset)
            game_id_offset += df['game_id'].nunique()  

            df.to_csv(output_csv, mode='a', header=first_file, index=False)
            first_file = False  

# Peco pelo PATH completo do diretorio em questao
input_folder = input("Entre com o dir, no formato (\"path/to/your/pgn/folder\"):\n")
if os.path.isdir(input_folder):
    print("Tudo certo no caminho ")
else:
    print(f"\"{input_folder}\" Não existe, ou está corrompido ")

output_file = "jogos_unificados.csv"
convert_folder_to_csv(input_folder, output_file)

