import pandas as pd

def _df_topic_coordinate(topic_coordinates):
    with open('./topic_coordinates.csv', 'w', encoding='utf-8') as f:
        f.write('topic,x,y,topics,cluster,Freq\n')
        for row in topic_coordinates:
            row_strf = ','.join((str(v) for v in row))
            f.write('%s\n' % row_strf)
    return pd.read_csv('./topic_coordinates.csv')

def _df_topic_info(topic_info):
    with open('./topic_info.csv', 'w', encoding='utf-8') as f:
        f.write('term,Category,Freq,Term,Total,loglift,logprob\n')
        for row in topic_info:
            row_strf = ','.join((str(v) for v in row))
            f.write('%s\n' % row_strf)
    return pd.read_csv('./topic_info.csv')

def _df_token_table(token_table):
    with open('./token_table.csv', 'w', encoding='utf-8') as f:
        f.write('term,Topic,Freq,Term\n')
        for row in token_table:
            row_strf = ','.join((str(v) for v in row))
            f.write('%s\n' % row_strf)
    return pd.read_csv('./token_table.csv')