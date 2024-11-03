import torch
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd


def get_data(df: pd.DataFrame):
    """
    Retrieve all required performance metrics data for cognitive diagnosis

    :param df: dataframe contaning all metrics for all models, and model names
    :return: (metric names, short metric names, concept names, Q matrix, logs dataframe, graph edges dataframe, graph
    adjacency matrix)
    """

    perf_metric_names = list(df.columns[1:])
    perf_metric_names_short = [
        'ACC',
        'PR0',
        'R0',
        'PR1',
        'R1',
        'BA',
        'FS0',
        'FS1',
        'AVGFS',
        'FM0',
        'FM1',
        'MKNS',
        'MCC',
        'JAC',
        'KAPPA'
    ]
    Q_matrix = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
    ])
    logs = []
    for student_id in tqdm(range(df.shape[0])):
        for exercise_id in range(len(perf_metric_names)):
            logs.append({'student_id': student_id, 'exercise_id': exercise_id, 'knowledge_emb': Q_matrix[exercise_id],
                         'score': df.iloc[student_id, exercise_id + 1]})
    df_logs = pd.DataFrame.from_records(logs)
    df_logs['score'] = df_logs['score'].astype('float32')
    assert np.min(df_logs['score']) == 0
    assert np.max(df_logs['score']) == 1

    concepts_list = ['C0', 'C1', 'BC', 'S0', 'S1', 'EQ', 'PR']
    know_graph_edges = [
        {'from': 'C0', 'to': 'S0'},
        {'from': 'C0', 'to': 'BC'},
        {'from': 'C1', 'to': 'S1'},
        {'from': 'C1', 'to': 'BC'},
        {'from': 'S0', 'to': 'EQ'},
        {'from': 'S0', 'to': 'PR'},
        {'from': 'S1', 'to': 'EQ'},
        {'from': 'S1', 'to': 'PR'},
    ]

    know_graph_adj = np.zeros((len(concepts_list), len(concepts_list)))
    for edge in know_graph_edges:
        i = concepts_list.index(edge['from'])
        j = concepts_list.index(edge['to'])
        edge['from'] = i
        edge['to'] = j
        know_graph_adj[i, j] = 1

    know_graph_edges_df = pd.DataFrame.from_records(know_graph_edges)

    return perf_metric_names, perf_metric_names_short, concepts_list, Q_matrix, df_logs, know_graph_edges_df, know_graph_adj


class CDMLogsDataset(torch.utils.data.Dataset):
    """Logs dataset for training CDM"""

    def __init__(self, df_logs):
        self.df_logs = df_logs

    def __len__(self):
        return self.df_logs.shape[0]

    def __getitem__(self, idx):
        row = self.df_logs.iloc[idx]
        return (row['student_id'], row['exercise_id']), row['score']