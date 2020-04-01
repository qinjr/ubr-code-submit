import pickle as pkl
import datetime
import time
import random
import sys
sys.path.append("..")
from elastic_client import *

RAW_DIR = '../../ubr4rec-data/ccmr/raw_data/'
FEATENG_DIR = '../../ubr4rec-data/ccmr/feateng_data/'

USER_NUM = 4920695
ITEM_NUM = 190129
DIRECTOR_NUM = 80171 + 1 #+1 no director
ACTOR_NUM = 213481 + 1 #+1 no actor
GENRE_NUM = 62 + 1 #+1 no genre
NATION_NUM = 1043 + 1 #+1 no nation

FEATURE_SIZE = 1 + 4920695 + 190129 + (80171 + 1) + (213481 + 1) + (62 + 1) + (1043 + 1) + 4

def get_season(month):
    if month >= 10:
        return 3
    elif month >= 7 and month <= 9:
        return 2
    elif month >= 4 and month <= 6:
        return 1
    else:
        return 0

def add_timestamp(raw_log_file, raw_log_ts_file):
    with open(raw_log_file, 'r') as f:
        newlines = []
        for line in f:
            uid, iid, rating, time_str = line[:-1].split(',')
            timestamp = str(int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple())))
            newlines.append(','.join([uid, iid, rating, time_str, timestamp]) + '\n')
    with open(raw_log_ts_file, 'w') as f:
        f.writelines(newlines)

def sort_raw_log(raw_log_ts_file, sorted_raw_log_ts_file):
    line_dict = {}
    with open(raw_log_ts_file) as f:
        for line in f:
            uid, iid, rating, time_str, ts = line[:-1].split(',')
            if uid not in line_dict:
                line_dict[uid] = [[line, int(ts)]]
            else:
                line_dict[uid].append([line, int(ts)])
            
    for uid in line_dict:
        line_dict[uid].sort(key = lambda x:x[1])
    print('sort complete')
    newlines = []
    for uid in line_dict:
        for tup in line_dict[uid]:
            newlines.append(tup[0])
    with open(sorted_raw_log_ts_file, 'w') as f:
        f.writelines(newlines)


def preprocess(raw_log_file, item_feat_dict_file, remapped_log_file, user_seq_dict_file):
    with open(item_feat_dict_file, 'rb') as f:
        item_feat_dict = pkl.load(f)
    
    with open(raw_log_file, 'r') as f:
        user_seq_dict = {}
        newlines = []
        for line in f:
            uid, iid, rating, time_str, timestamp = line[:-1].split(',')
            uid = str(int(uid) + 1)
            iid = str(int(iid) + 1 + USER_NUM)
            item_feat = list(map(str, item_feat_dict[iid]))
            
            month = int(time_str.split('-')[1])
            sea_id = str(get_season(month) + 1 + 4920695 + 190129 + (80171 + 1) + (213481 + 1) + (62 + 1) + (1043 + 1))

            newline = ','.join([uid, iid] + item_feat + [sea_id, str(timestamp)]) + '\n'
            newlines.append(newline)

            if uid not in user_seq_dict:
                user_seq_dict[uid] = [iid]
            else:
                user_seq_dict[uid].append(iid)
            
    
    with open(remapped_log_file, 'w') as f:
        f.writelines(newlines)
        print('remapped log file completed')
    
    with open(user_seq_dict_file, 'wb') as f:
        pkl.dump(user_seq_dict, f)

def random_sample(min = 4920696, max = 4920696 + 190128):
    return str(random.randint(min, max))

def neg_sample(uid, user_neg_dict):
    if uid in user_neg_dict:
        return str(random.choice(user_neg_dict[uid]))
    else:
        return random_sample()

def gen_target_seq(input_file,
                    user_neg_dict_file,
                    user_seq_dict_file,
                    target_train_file, 
                    target_vali_file, 
                    target_test_file, 
                    user_seq_file, 
                    database_file,
                    context_dict_train_file, 
                    context_dict_vali_file, 
                    context_dict_test_file):
    with open(user_seq_dict_file, 'rb') as f:
        user_seq_dict = pkl.load(f)
    with open(user_neg_dict_file, 'rb') as f:
        user_neg_dict = pkl.load(f)
    
    line_dict = {}
    context_dict_train = {}
    context_dict_vali = {}
    context_dict_test = {}

    with open(input_file, 'r') as f:
        for line in f:
            uid, iid, did, aid, gid, nid, sea_id, time_stamp = line[:-1].split(',')
            if uid not in line_dict:
                line_dict[uid] = [line]
            else:
                line_dict[uid].append(line)        

        target_train_lines = []
        target_vali_lines = []
        target_test_lines = []
        user_seq_lines = []
        database_lines = []
        
        for uid in user_seq_dict:
            if len(user_seq_dict[uid]) > 3:
                target_train_lines += [','.join([uid, user_seq_dict[uid][-3]]) + '\n']
                target_train_lines += [','.join([uid, neg_sample(uid, user_neg_dict)]) + '\n']
                context_dict_train[uid] = [int(line_dict[uid][-3][:-1].split(',')[-2])]

                target_vali_lines += [','.join([uid, user_seq_dict[uid][-2]]) + '\n']
                target_vali_lines += [','.join([uid, neg_sample(uid, user_neg_dict)]) + '\n']
                context_dict_vali[uid] = [int(line_dict[uid][-2][:-1].split(',')[-2])]

                target_test_lines += [','.join([uid, user_seq_dict[uid][-1]]) + '\n']
                target_test_lines += [','.join([uid, neg_sample(uid, user_neg_dict)]) + '\n']
                context_dict_test[uid] = [int(line_dict[uid][-1][:-1].split(',')[-2])]
                
                user_seq = user_seq_dict[uid][:-3]
                user_seq_lines += [','.join(user_seq) + '\n'] * 2 #(1 pos and 1 neg item)
                
                database_lines += line_dict[uid][:-3]
        
        with open(target_train_file, 'w') as f:
            f.writelines(target_train_lines)
        with open(target_vali_file, 'w') as f:
            f.writelines(target_vali_lines)
        with open(target_test_file, 'w') as f:
            f.writelines(target_test_lines)
        
        with open(user_seq_file, 'w') as f:
            f.writelines(user_seq_lines)
        with open(database_file, 'w') as f:
            f.writelines(database_lines)
        
        with open(context_dict_train_file, 'wb') as f:
            pkl.dump(context_dict_train, f)
        with open(context_dict_vali_file, 'wb') as f:
            pkl.dump(context_dict_vali, f)
        with open(context_dict_test_file, 'wb') as f:
            pkl.dump(context_dict_test, f)


def insert_elastic(input_file):      
    writer = ESWriter(input_file, 'ccmr')
    writer.write()

if __name__ == '__main__':
    # add_timestamp(FEATENG_DIR + 'rating_pos.csv', FEATENG_DIR + 'rating_pos_ts.csv')
    # sort_raw_log(FEATENG_DIR + 'rating_pos_ts.csv', FEATENG_DIR + 'sorted_rating_pos_ts.csv')

    # preprocess(FEATENG_DIR + 'sorted_rating_pos_ts.csv', FEATENG_DIR + 'item_feat_dict.pkl', FEATENG_DIR + 'remapped_log.csv', FEATENG_DIR + 'user_seq_dict.pkl')
    # gen_target_seq(FEATENG_DIR + 'remapped_log.csv', FEATENG_DIR + 'user_neg_dict.pkl', FEATENG_DIR + 'user_seq_dict.pkl', FEATENG_DIR + 'target_train.txt', FEATENG_DIR + 'target_vali.txt', FEATENG_DIR + 'target_test.txt', FEATENG_DIR + 'user_seq.txt', FEATENG_DIR + 'database.txt', 
    #                 FEATENG_DIR + 'context_dict_train.pkl', FEATENG_DIR + 'context_dict_vali.pkl', FEATENG_DIR + 'context_dict_test.pkl')
    insert_elastic(FEATENG_DIR + 'database.txt')



