# -*- coding: utf-8 -*-

import pandas as pd
import os
import random
import numpy as np
import gc


#############################
# 步骤:
# 1. 清理所有原始数据
# 2. 预处理各侧原始数据
# 3. 各侧生成自己所需中间数据
# 4. 各侧生成自己特征
# 5. 合并特征
# 6. 生成送入模型所需数据格式
# 7. 划分数据集
#############################

#############################
# 辅助函数

def read_file(*filename):
    ret = [ pd.io.parsers.read_csv(file) for file in filename ]
    return ret


def write_data(df,filename):
    print("开始向 %s 写入数据" % filename)
    df.to_csv(filename,index=False)


# 把长信息展开为宽信息
def doc_expend(filename,index_name):
    print("开始轴向旋转文件: %s" % filename)
    df = pd.io.parsers.read_csv(filename)
    df = df.drop_duplicates(['document_id',index_name])
    df = df.pivot('document_id',index_name)['confidence_level']
    df = df.fillna(0)
    print("完成文件 %s 的轴向旋转" % filename)
    return df


# 批量计算余弦相似度
def cos_similar_batch(arr1, arr2):
    arr_product = arr1 * arr2
    arr_sum = arr_product.sum(1)
    norm1 = np.linalg.norm(arr1,axis=1)
    norm2 = np.linalg.norm(arr2,axis=1)
    norm_product = norm1 * norm2
    ret = arr_sum / norm_product
    # print('arr_sum:')
    # print(arr_sum)
    # print('norm_product')
    # print(norm_product)
    # print('ret')
    # print(ret)
    return ret


# 计算文档相似度
def calculate_doc_similar(events, meta, col_name):
    cols = meta.columns[1:]
    view_cols = [n+'_conf_view' for n in cols]
    ad_cols = [n+'_conf_ad' for n in cols]

    tmp = pd.merge(events,meta,left_on='document_id_view',right_on='document_id',how='inner')
    tmp = pd.merge(tmp,meta,left_on='document_id_ad',right_on='document_id',how='inner',suffixes=['_conf_view','_conf_ad'])
    print("tmp original rows:%d"%len(tmp))
    #tmp = tmp.drop_duplicates(['document_id_view','document_id_ad'])
    #print("tmp deduplicated rows:%d"%len(tmp))

    views = np.asarray(tmp[view_cols])
    ads = np.asarray(tmp[ad_cols])

    cos_sim = cos_similar_batch(views,ads)

    new_dt = pd.DataFrame()
    new_dt[['document_id_view','document_id_ad']] = tmp[['document_id_view','document_id_ad']]
    new_dt[col_name] = cos_sim
    # print(new_dt)
    return new_dt


# 读取数据并生成相似度特征,写入文件
def make_similar(dir, data_filename, meta_filename, col_name, out_filename, chunk_size):
    #chunk_size = 10000
    save_point = 1000000
    cnt = 1
    first_line = True

    data_path = os.path.join(dir,data_filename)
    meta_path = os.path.join(dir,meta_filename)
    out_path = os.path.join(dir,out_filename)

    if os.path.exists(out_path) :
        os.remove(out_path)

    print('读取文件%s,生成%s'%(meta_filename,col_name))
    dt_iter = pd.io.parsers.read_csv(data_path,chunksize=chunk_size,usecols=['document_id_view','document_id_ad'])
    meta = pd.io.parsers.read_csv(meta_path)

    ret = None
    for i,dt in enumerate(dt_iter):
        # print(dt)
        new_dt = calculate_doc_similar(dt,meta,col_name)
        new_merge = pd.merge(dt, new_dt,on=['document_id_view','document_id_ad'],how='left')
        # new_merge = new_dt
        new_merge[col_name].fillna(0)

        if ret is not None :
            ret = pd.concat([ret, new_merge])
        else:
            ret = new_merge

        if i * chunk_size % save_point == 0:
            print('正在写出第%d份记录到文件%s'%(cnt,out_path))
            if first_line :
                ret.to_csv(out_path, columns=['document_id_view', 'document_id_ad', col_name], index=False, mode='a')
                first_line = False
            else:
                ret.to_csv(out_path, columns=['document_id_view', 'document_id_ad', col_name], index=False, mode='a', header=False)

            ret = None
            cnt += 1
    if ret is not None :
        print('正在写出第%d份记录到文件%s'%(cnt,out_path))
        if first_line:
            ret.to_csv(out_path, columns=['document_id_view', 'document_id_ad', col_name], index=False, mode='a')
        else:
            ret.to_csv(out_path, columns=['document_id_view', 'document_id_ad', col_name], index=False, mode='a', header=False)

    print('完成%s的创建'%col_name)


# 将新的特征加入到训练数据表中,原数据表与新feature表格应该具有同样排列的数据
# table1 是较大的数据文件
# table2 是较小的新特征文件
def merge_feature(dir, table1, table2, out, chunk_size=10000000):
    file1 = os.path.join(dir, table1)
    file2 = os.path.join(dir, table2)
    file_out = os.path.join(dir, out)
    first_line = True

    print('开始合并%s和%s' % (table1,table2))

    data2 = pd.io.parsers.read_csv(file2)

    if os.path.exists(file_out):
        os.remove(file_out)

    data1_iter = pd.io.parsers.read_csv(file1, chunksize=chunk_size)

    for i,data1 in enumerate(data1_iter):
        print('处理第%i个chunk' % i)
        data_out = pd.merge(data1, data2, how='left')
        if first_line:
            data_out.to_csv(file_out, index=False)
            first_line = False
        else:
            data_out.to_csv(file_out, index=False, mode='a', header=False)

    print('合并%s和%s完成' % (table1,table2))


###################################################################
# 数据读取


def read_clicks_train(dir):
    file = os.path.join(dir, 'clicks_train.csv')
    dtypes = {
        'display_id': np.int64,
        'ad_id': np.int64,
        'clicked': np.int
    }

    dt = pd.read_csv(file, ',', dtype=dtypes)
    return dt

def read_clicks_test(dir):
    file = os.path.join(dir, 'clicks_test.csv')
    dtypes = {
        'display_id': np.int64,
        'ad_id': np.int64,
    }

    dt = pd.read_csv(file, ',', dtype=dtypes)
    return dt

def read_events(dir):
    file = os.path.join(dir, 'events.csv')
    dtypes = {
        'display_id': np.int64,
        'uuid': np.str,
        'document_id': np.int64,
        'timestamp': np.int64,
        'platform': np.str,
        'geo_location': np.str
    }

    dt = pd.read_csv(file, ',', dtype=dtypes, na_values='NaN')
    dt['platform'] = pd.to_numeric(dt['platform'],errors='coerce').fillna(0).astype(np.int64)

    return dt

def read_promoted_content(dir):
    file = os.path.join(dir, 'promoted_content.csv')
    dtypes = {
        'ad_id': np.int64,
        'document_id': np.int64,
        'campaign_id': np.int64,
        'advertiser_id': np.int64
    }

    dt = pd.read_csv(file, ',', dtype=dtypes)
    return dt

def read_documents_meta(dir):
    file = os.path.join(dir, 'documents_meta.csv')
    dtypes = {
        'document_id': np.float,
        'source_id': np.float,
        'publisher_id': np.float,
        'publish_time': np.str,
    }

    dt = pd.read_csv(file, ',', dtype=dtypes, na_values='NaN')

    int_cols = ['document_id', 'source_id', 'publisher_id']
    for col in int_cols:
        dt[col] = pd.to_numeric(dt[col],errors='coerce').fillna(0).astype(np.int64)
    return dt


###################################################################
# 数据清洗和预处理


# 清洗event数据
def clean_event(df):
    print('清洗event数据..................')
    print('event data size before dropna: %i' % len(df))
    not_none_col = ['display_id', 'document_id', 'uuid']
    df.dropna(subset=not_none_col, inplace=True)
    print('event data size after dropna: %i' % len(df))

    fillna_dict = {
        'timestamp': df['timestamp'].median(),
        'platform': 0,
        'geo_location': 'UNK',
    }
    df.fillna(fillna_dict, inplace=True)
    return df


# 清洗promoted数据
def clean_promoted(df):
    print('清洗promoted数据..................')
    print('promoted data size before dropna: %i' % len(df))
    not_none_col = ['ad_id', 'document_id']
    df.dropna(subset=not_none_col, inplace=True)
    print('promoted data size after dropna: %i' % len(df))


    fillna_dict = {
        'campaign_id': df['campaign_id'].mode().iloc[0],
        'advertiser_id': df['advertiser_id'].mode().iloc[0],
    }
    df.fillna(fillna_dict, inplace=True)
    return df


# 清洗documents数据
def clean_documents(df):
    print('清洗documents数据..................')
    print('documents data size before dropna: %i' % len(df))
    not_none_col = ['document_id']
    df.dropna(subset=not_none_col, inplace=True)
    print('documents data size after dropna: %i' % len(df))


    fillna_dict = {
        'source_id': df['source_id'].mode().iloc[0],
        'publisher_id': df['publisher_id'].mode().iloc[0],
        'publish_time': df['publish_time'].mode().iloc[0]
    }
    df.fillna(fillna_dict, inplace=True)
    return df


# 预处理event数据
# 拆分地理位置信息
def preprocess_event(df):
    print('预处理event数据..................')
    geo = df["geo_location"]
    geo_new = pd.DataFrame([x.split(">") for x in geo.astype('string')],columns=["geo_0","geo_1","geo_2"], dtype="string")
    df = df.join(geo_new,rsuffix="geo_")
    df.drop(["geo_location"],1,inplace=True)

    df['platform'] = pd.to_numeric(df['platform'],errors='coerce').fillna(0).astype('int')
    df.fillna({'geo_0':'UNK','geo_1':'UNK','geo_2':'UNK'},inplace=True)
    df[df['geo_0'] == 'nan'] = 'UNK'
    return df

################################################
# 生成中间数据


def _generate_document_pair(dir):
    outfile = os.path.join(dir, 'document_pair.csv')
    tmp_outfile = os.path.join(dir, 'tmp_document_pair.csv')
    first_line = True

    click_train = pd.io.parsers.read_csv(os.path.join(dir, 'clicks_train.csv'), usecols=['display_id', 'ad_id'], chunksize=1000000)
    click_test = pd.io.parsers.read_csv(os.path.join(dir, 'clicks_test.csv'), usecols=['display_id', 'ad_id'], chunksize=1000000)

    events = pd.read_csv(os.path.join(dir, 'events.csv'), usecols=['display_id', 'document_id'])
    ads = pd.read_csv(os.path.join(dir, 'promoted_content.csv'), usecols=['ad_id', 'document_id'])

    if os.path.exists(tmp_outfile):
        os.remove(tmp_outfile)

    for i, click_slice in enumerate(click_train):
        print('处理第%i块clicks_train.csv记录' % i)
        c_e = pd.merge(click_slice, events, how='left')
        c_e_p = pd.merge(c_e, ads, on='ad_id', suffixes=['_view', '_ad'])
        c_e_p.drop_duplicates(['document_id_view', 'document_id_ad'], inplace=True)
        if first_line:
            c_e_p.to_csv(tmp_outfile, index=False, mode='a', columns=['document_id_view', 'document_id_ad'])
            first_line = False
        else:
            c_e_p.to_csv(tmp_outfile, index=False, mode='a', columns=['document_id_view', 'document_id_ad'], header=False)

    for i, click_slice in enumerate(click_test):
        print('处理第%i块clicks_test.csv记录' % i)
        c_e = pd.merge(click_slice, events, how='left')
        c_e_p = pd.merge(c_e, ads, on='ad_id', suffixes=['_view', '_ad'])
        c_e_p.drop_duplicates(['document_id_view', 'document_id_ad'], inplace=True)
        c_e_p.to_csv(tmp_outfile, index=False, mode='a', header=False, columns=['document_id_view', 'document_id_ad'])

    print('正在清除文档对中的重复项')
    del c_e,c_e_p
    pairs = pd.read_csv(tmp_outfile)
    print('去重前共有%i条记录'%len(pairs))
    pairs.drop_duplicates(inplace=True)
    print('去重后共有%i条记录' % len(pairs))

    pairs.to_csv(outfile, index=False)


def generate_document_tmp_data(dir):
    print('正在生成document中间数据')

    print('正在展开长数据')
    args = [
        {
            'filename': os.path.join(dir, 'documents_categories.csv'),
            'col_name': 'category_id',
            'outname': os.path.join(dir, 'documents_categories_ex.csv'),
        },
        {
            'filename': os.path.join(dir, 'documents_topics.csv'),
            'col_name': 'topic_id',
            'outname': os.path.join(dir, 'documents_topics_ex.csv'),
        },
    ]

    for arg in args:
        df = doc_expend(arg['filename'],arg['col_name'])
        df.to_csv(arg['outname'])

    print('生成需要的文档对')
    _generate_document_pair(dir)


##################################################
# 生成特征

# 生成event侧特征
def generate_event_feature(dir):
    print('正在生成event特征')

    events = pd.read_csv(os.path.join(dir, 'events_processed.csv'))
    events.rename(columns={'document_id':'document_id_view'}, inplace=True)
    events.to_csv(os.path.join(dir,'feature_events.csv'), index=False)

# 生成document侧特征
def generate_document_feature(dir):
    print('正在生成document特征')

    document_ids = pd.read_csv(os.path.join(dir, 'events_processed.csv'), usecols=['document_id'])
    document_ids.drop_duplicates()
    documents_meta = pd.read_csv(os.path.join(dir, 'documents_meta_processed.csv'))

    merged = pd.merge(document_ids, documents_meta)
    merged.drop_duplicates(['document_id'], inplace=True)

    fillna_dict = {
        'source_id': documents_meta['source_id'].mode().iloc[0],
        'publisher_id': documents_meta['publisher_id'].mode().iloc[0],
        'publish_time': documents_meta['publish_time'].mode().iloc[0],
    }

    cols_rename = {
        'document_id' : 'document_id_view',
        'source_id' : 'source_id_view',
        'publisher_id' : 'publisher_id_view',
        'publish_time' : 'publish_time_view',
    }
    merged.rename(columns=cols_rename, inplace=True)

    merged.fillna(fillna_dict, inplace=True)
    merged.to_csv(os.path.join(dir, 'feature_documents.csv'), index=False)


# 生成ad侧特征
def generate_promoted_feature(dir):
    print('正在生成ad特征')

    # click_train = pd.io.parsers.read_csv(os.path.join(dir, 'clicks_train.csv'), usecols=['ad_id'])
    # click_test = pd.io.parsers.read_csv(os.path.join(dir, 'clicks_test.csv'), usecols=['ad_id'])
    # click = pd.concat([click_train, click_test])
    # del click_train, click_test
    # click.drop_duplicates()

    promoted = pd.read_csv(os.path.join(dir, 'promoted_content_processed.csv'))
    documents_meta = pd.read_csv(os.path.join(dir, 'documents_meta_processed.csv'))

    merged = pd.merge(promoted, documents_meta, how='left')

    fillna_dict = {
        'source_id': documents_meta['source_id'].mode().iloc[0],
        'publisher_id': documents_meta['publisher_id'].mode().iloc[0],
        'publish_time': documents_meta['publish_time'].mode().iloc[0]
    }
    merged.fillna(fillna_dict, inplace=True)
    cols_rename = {
        'document_id' : 'document_id_ad',
        'source_id' : 'source_id_ad',
        'publisher_id' : 'publisher_id_ad',
        'publish_time' : 'publish_time_ad',
    }
    merged.rename(columns=cols_rename, inplace=True)
    merged.to_csv(os.path.join(dir, 'feature_promoted.csv'), index=False)


# 生成document-ad特征
def generate_document_promoted_feature(dir):
    print('生成document-ad特征')

    print('正在生成document类别相似度')
    make_similar('./data','document_pair.csv','documents_categories_ex.csv','category_similar','tmp_category_similar.csv',100000)
    print('正在生成document主题相似度')
    make_similar('./data','document_pair.csv','documents_topics_ex.csv','topic_similar','tmp_topic_similar.csv',10000)

    print('合并document-ad特征')
    merge_feature('./data', 'tmp_topic_similar.csv', 'tmp_category_similar.csv', 'feature_document_ad.csv', 100000)


################################################
# 合并所有特征

def merge_features(dir):
    print('合并各部分特征')
    merge_feature('./data', 'clicks_train.csv', 'feature_events.csv', 'tmp_click_train_events.csv', 200000)
    gc.collect()
    merge_feature('./data', 'tmp_click_train_events.csv', 'feature_documents.csv', 'tmp_click_train_events_documents.csv', 1000000)
    gc.collect()
    merge_feature('./data', 'tmp_click_train_events_documents.csv', 'feature_promoted.csv', 'tmp_click_train_events_documents_promoted.csv', 1000000)
    gc.collect()
    merge_feature('./data', 'tmp_click_train_events_documents_promoted.csv', 'feature_document_ad.csv', 'feature_train.csv', 1000000)
    gc.collect()

    merge_feature('./data', 'clicks_test.csv', 'feature_events.csv', 'tmp_click_test_events.csv', 1000000)
    gc.collect()
    merge_feature('./data', 'tmp_click_test_events.csv', 'feature_documents.csv', 'tmp_click_test_events_documents.csv', 1000000)
    gc.collect()
    merge_feature('./data', 'tmp_click_test_events_documents.csv', 'feature_promoted.csv', 'tmp_click_test_events_documents_promoted.csv', 1000000)
    gc.collect()
    merge_feature('./data', 'tmp_click_test_events_documents_promoted.csv', 'feature_document_ad.csv', 'feature_test.csv', 1000000)
    gc.collect()

################################################
# 划分数据集

def split_data(dir, feature_file, out_file, bins=5):
    file = os.path.join(dir, feature_file)
    tmp_name = out_file.split('.')
    out_files = [ os.path.join(dir, '.'.join( tmp_name[0:1] + [str(i)] + tmp_name[1:] )) for i in xrange(bins) ]

    def _split_index(index_range):
        index_list = range(index_range)
        random.shuffle(index_list)
        interval = int(len(index_list) / bins)
        index_bins = [ index_list[i*interval: (i+1)*interval] for i in xrange(bins) ]
        return index_bins

    df_iter = pd.read_csv(file, chunksize=1000000)

    first_line = True
    for out_file in out_files:
        if os.path.exists(out_file):
            os.remove(out_file)

    for df in df_iter:
        samples = []

        df_p = df[df['clicked'] == 1]
        p_cnt = len(df_p)
        print('划分%i个正样本' % p_cnt)
        p_bins = _split_index(p_cnt)
        for i, bin in enumerate(p_bins):
            p_sample = df.iloc[bin]
            if len(samples) <= i:
                samples.append(p_sample)
            else:
                samples[i] = pd.concat([samples[i], p_sample])
            print('p_samples[%i] size:%i' %(i,len(samples[i])))


        df_n = df[df['clicked'] == 0]
        n_cnt = len(df_n)
        print('划分%i个负样本' % n_cnt)
        n_bins = _split_index(n_cnt)
        for i, bin in enumerate(n_bins):
            n_sample = df.iloc[bin]
            if len(samples) <= i:
                samples.append(n_sample)
            else:
                samples[i] = pd.concat([samples[i], n_sample])
            print('all_samples[%i] size:%i' %(i,len(samples[i])))

        for i, sample in enumerate(samples):
            sample = sample.sample(frac=1).reset_index(drop=True)
            print('sample %i size:%i' %(i, len(sample)))
            print('out file:%s' % out_files[i])
            if first_line:
                print('正在向%s写入首行数据'% out_files[i])
                sample.to_csv(out_files[i], index=False, mode='a')
            else:
                print('正在向%s写入后续数据'% out_files[i])
                sample.to_csv(out_files[i], index=False, mode='a', header=False)
        first_line = False





################################################
# 输出符合tensorflow格式的特征

def _read_feature_des(dir, des_filename):
    des_file = os.path.join(dir, des_filename)
    des = pd.read_csv(des_file,';')

    col_with_label = des[des['enable']]['name'].values.tolist()
    col_with_label.sort()

    col_without_label = des[des['enable']][des['type'] != 'label']['name'].values.tolist()
    col_without_label.sort()

    return col_with_label, col_without_label

def write_for_tf(dir, filename, outname, cols):
    in_file = os.path.join(dir, filename)
    out_file = os.path.join(dir, outname)

    if os.path.exists(out_file):
        os.remove(out_file)


    chunk_size = 10000000

    data_iter = pd.io.parsers.read_csv(in_file, usecols=cols, chunksize=chunk_size)

    for (i,data) in enumerate(data_iter):
        print('正在转写第%i块记录' % i)
        data.to_csv(out_file, mode='a', index=False, header=False, columns=cols)

    print("finished writing")


def generate_feeding_data(dir, train_files=None, test_files=None):
    cols_train, cols_test = _read_feature_des(dir,'feature_des.csv')

    if train_files:
        print('输出训练数据')
        for file in train_files:
            write_for_tf(dir, file, 'tf_'+file, cols_train)

    if test_files:
        print('输出预测数据')
        for file in test_files:
            write_for_tf(dir, file, 'tf_'+file, cols_test)



################################################
# 完整流程

def data_process(dir):
    # step1. 读取数据和数据清洗/预处理
    # print('开始预处理events数据')
    # events = read_events(dir)
    # events_processed = clean_event(events)
    # events_processed = preprocess_event(events_processed)
    # events_processed.to_csv(os.path.join(dir, 'events_processed.csv'), index=False)
    # del events, events_processed
    # print('预处理events数据完毕')

    # print('开始预处理documents_meta数据')
    # documents = read_documents_meta(dir)
    # documents_processed = clean_documents(documents)
    # documents_processed.to_csv(os.path.join(dir, 'documents_meta_processed.csv'), index=False)
    # del documents, documents_processed
    # print('预处理documents_meta数据完毕')
    #
    # print('开始预处理promoted_content数据')
    # promoted = read_promoted_content(dir)
    # promoted_processed = clean_promoted(promoted)
    # promoted_processed.to_csv(os.path.join(dir, 'promoted_content_processed.csv'), index=False)
    # del promoted, promoted_processed
    # print('预处理promoted_content数据完毕')

    # # step2. 生成中间数据
    # generate_document_tmp_data(dir)

    # # step3. 生成各部分数据
    # generate_event_feature(dir)
    # generate_document_feature(dir)
    # generate_document_promoted_feature(dir)
    # generate_promoted_feature(dir)

    # # step4. 合并各特征
    # merge_features(dir)

    # # step5. 划分数据集
    # split_data(dir, 'feature_train.csv', 'feature_train_splited.csv')

    # step6. 转写数据集
    generate_feeding_data(dir, ['feature_train_splited.0.csv','feature_train_splited.1.csv','feature_train_splited.2.csv','feature_train_splited.3.csv','feature_train_splited.4.csv'])

if __name__ == '__main__':
    data_process('./data')











