# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import pandas as pd
import json
import numpy as np
import math
from profile_hook import ProfilerHook
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)

HIDDEN_UNITS = [1024,512,256]
TRAIN_FILELIST = ["data_sample.csv"]
VALID_FILELIST = ["data_sample.csv"]
DES_FILENAME = "data_sample_des.csv"

def make_filepath(dir, files):
    return map(lambda file: os.path.join(dir,file), files)

def read_data(filenames, col_names):
    data = None
    for f in filenames:
        this_data = pd.read_csv(f, sep=',', usecols=col_names)
        print('this data: %i' % this_data.size)
        if not (data is None):
            data = pd.concat([data, this_data])
        else:
            data = this_data
    return data

class WDL(object):
    def __init__(self, dir, model_dir,model_type,batch_size,train_epoches,dnn_lr,hidden_units,des_file,cross_des_file = None, data=None):
        self.data = data
        self.dir = dir
        self.model_dir = os.path.join(dir, model_dir)
        self.model_type = model_type
        self.batch_size = batch_size
        self.train_epoches = train_epoches
        # self.train_files = train_files
        # self.valid_files = valid_files
        self.des_file = os.path.join(dir, des_file)
        self.cross_des_file = os.path.join(dir, cross_des_file)
        self.hidden_units = hidden_units
        self.eval_sample_interval = 50000000
        self.save_ckpt_interval = 1000000
        self.save_summary_interval = 1000000
        self.enqueue_sample_num = 10000000
        self.valid_sum = 17000000
        self.valid_batch_size = 300
        self.dnn_lr = dnn_lr
        self.write_params()

        #self.read_col_des(self.des_file)
        self.read_feature_des(self.des_file, self.cross_des_file)
        self.get_cols()
        self.build_estimator()


    def write_params(self):
        params = {
            "modle_type" : self.model_type,
            "batch_size" : self.batch_size,
            "train_epoches" : self.train_epoches,
            # "train_files" : self.train_files,
            # "valid_files" : self.valid_files,
            "des_file" : self.des_file,
            "hidden_units" : self.hidden_units,
            "eval_sample_interval" : self.eval_sample_interval,
            "lr" : self.dnn_lr
        }
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        filename = os.path.join(self.model_dir,'meta_params')
        with open(filename,'w') as f:
            for (key,value) in params.items():
                f.write("\t".join([key,str(value)])+'\n')

    # 读取特征描述文件
    def read_feature_des(self, filename, cross_filename=None):

        def _convert_val(value):
            val = value[0]
            dtype = value[1]
            if dtype == 'int32':
                return [int(val)]
            elif dtype == 'float32':
                return [float(val)]
            else:
                return [val]


        df = pd.read_csv(filename, sep=';')
        print('正在解析%i个特征描述' % len(df))
        print(df)

        self.col_des = df[df['enable']]
        self.label_name = self.col_des[self.col_des['type'] == 'label']['name'].values[0]

        cols = self.col_des

        self.col_name_sorted_with_label = cols['name'].values.tolist()
        self.col_name_sorted_with_label.sort()

        default_val = [cols[cols['name'] == k]['default'].iloc[0] for k in self.col_name_sorted_with_label]
        default_dtype = [cols[cols['name']==k]['dtype'].iloc[0] for k in self.col_name_sorted_with_label]
        self.defaults_with_label = map(_convert_val,zip(default_val,default_dtype))
        print('col_name_sorted:')
        print(self.col_name_sorted_with_label)
        print('defaults:')
        print(self.defaults_with_label)

        self.col_name_sorted_without_label = cols[cols['type']!='label']['name'].values.tolist()
        self.col_name_sorted_without_label.sort()
        default_val_without_label = [cols[cols['name'] == k]['default'].iloc[0] for k in self.col_name_sorted_without_label]
        default_dtype_without_label = [cols[cols['name']==k]['dtype'].iloc[0] for k in self.col_name_sorted_without_label]
        self.defaults_without_label = map(_convert_val,zip(default_val_without_label,default_dtype_without_label))

        self.cross_des = None
        if cross_filename:
            cross_df = pd.read_csv(cross_filename, sep=';')
            self.cross_des = cross_df[cross_df['enable']]


    def get_cols(self):
        layers = tf.contrib.layers
        self.wide_cols = []
        self.deep_cols = []
        col_des = self.col_des[self.col_des['type'] != 'label']
        col_dict = {}

        for i in xrange(len(col_des)):
            feature = col_des.iloc[i]

            if feature['method'] == 'feed':
                col = layers.real_valued_column(column_name=feature['name'], default_value=float(feature['default']))
            elif feature['method'] == 'indexed':
                col = layers.sparse_column_with_integerized_feature(column_name=feature['name'],bucket_size=int(feature["param"]))
            elif feature['method'] == 'hash':
                col = layers.sparse_column_with_hash_bucket(column_name=feature['name'], hash_bucket_size=int(feature['param']), combiner='sqrtn' )

            if feature['model'] == 'wide' or feature['model'] == 'wdl':
                col_wide = col
                if feature['method'] == 'feed':
                    col_wide = layers.bucketized_column(col, json.loads(feature['param']))
                self.wide_cols.append(col_wide)
                col_dict[feature['name']] = col_wide
                print(col_dict)

            if feature['model'] == 'deep' or feature['model'] == 'wdl':
                col_deep = col
                if feature['method'] != 'feed':
                    col_deep = layers.embedding_column(col_deep, dimension=feature["embedding_size"])
                self.deep_cols.append(col_deep)
                if not col_dict.has_key(feature['name']):
                    col_dict[feature['name']] = col_deep


            # if feature['model'] == 'wide' or feature['model'] == 'wdl':
            #
            #     if feature['method'] == 'feed':
            #         col = layers.real_valued_column(column_name=feature['name'], default_value=float(feature['default']))
            #         col = layers.bucketized_column(col, json.loads(feature['param']))
            #     elif feature['method'] == 'indexed':
            #         col = layers.sparse_column_with_integerized_feature(column_name=feature['name'],bucket_size=int(feature["param"]))
            #     elif feature['method'] == 'hash':
            #         col = layers.sparse_column_with_hash_bucket(column_name=feature['name'], hash_bucket_size=int(feature['param']), combiner='sqrtn' )
            #
            #     self.wide_cols.append(col)
            #     col_dict[feature['name']] = col
            #
            # if feature['model'] == 'deep' or feature['model'] == 'wdl':
            #     if feature['method'] == 'feed':
            #         col = layers.real_valued_column(column_name=feature['name'], default_value=float(feature['default']))
            #     elif feature['method'] == 'indexed':
            #         col = layers.embedding_column(layers.sparse_column_with_integerized_feature(column_name=feature['name'],bucket_size=int(feature["param"])),dimension=feature["embedding_size"])
            #     elif feature['method'] == 'hash':
            #         col = layers.embedding_column(layers.sparse_column_with_hash_bucket(column_name=feature['name'], hash_bucket_size=int(feature['param']), combiner='sqrtn'),dimension=feature["embedding_size"])
            #
            #     self.deep_cols.append(col)
            #     if not col_dict.has_key(feature['name']):
            #         col_dict[feature['name']] = col

        if self.cross_des is not None:
            for i in xrange(len(self.cross_des)):
                feature = self.cross_des.iloc[i]
                col = layers.crossed_column([col_dict[feature['feature1']], col_dict[feature['feature2']]], hash_bucket_size=int(feature['size']), combiner='sum')
                self.wide_cols.append(col)
                print(col)


    # 建立预测器
    def build_estimator(self):
        """Build an estimator."""
        save_ckpt = int(self.save_ckpt_interval / self.batch_size)
        save_summary = int(self.save_summary_interval / self.batch_size)

        if self.model_type == "wide":
            self.model = tf.contrib.learn.LinearClassifier(
                model_dir=self.model_dir,
                feature_columns=self.wide_cols,
                config=tf.contrib.learn.RunConfig(gpu_memory_fraction = 1, log_device_placement=False,save_summary_steps=save_summary, save_checkpoints_steps=save_ckpt,save_checkpoints_secs=None)
            )
        elif self.model_type == "deep":
            self.model = tf.contrib.learn.DNNClassifier(
                model_dir=self.model_dir,
                feature_columns=self.deep_cols,
                hidden_units=self.hidden_units,
                optimizer=tf.train.AdagradOptimizer(self.dnn_lr),
                config=tf.contrib.learn.RunConfig(gpu_memory_fraction = 1, log_device_placement=False,save_summary_steps=save_summary, save_checkpoints_steps=save_ckpt,save_checkpoints_secs=None)
            )
        else:
            self.model = tf.contrib.learn.DNNLinearCombinedClassifier(
                model_dir=self.model_dir,
                linear_feature_columns=self.wide_cols,
                dnn_feature_columns=self.deep_cols,
                dnn_hidden_units=self.hidden_units,
                dnn_optimizer=tf.train.AdagradOptimizer(self.dnn_lr),
                config=tf.contrib.learn.RunConfig(gpu_memory_fraction = 0.6, log_device_placement=False,save_summary_steps=save_summary, save_checkpoints_steps=save_ckpt,save_checkpoints_secs=None)
            )
        return


    # 数据读取操作符
    def read_my_file_format(self,filename_queue,col_names, defaults):
        reader = tf.TextLineReader()
        key, record_string = reader.read_up_to(filename_queue, self.enqueue_sample_num)

        print('record_string:')
        print(record_string)
        print('defaults:')
        print(defaults)

        cols = tf.decode_csv(record_string, record_defaults = defaults, name = 'decode_csv')

        #cols = tf.decode_csv(record_string )
        record = dict([(col_names[i],tf.expand_dims(cols[i],1)) for i in xrange(len(cols))])
        print('records:')
        print(record)

        return record


    # 数据预处理
    def preprocess(self, record, labeled):
        if not labeled :
            features = record
            return features

        col_name = self.col_des[self.col_des['type'] != 'label']['name'].values.tolist()
        label_name = self.col_des[self.col_des['type'] == 'label']['name'].values.tolist()[0]


        features = dict([(col_name[i], record[col_name[i]]) for i in xrange(len(col_name)) ])
        labels = {label_name:record[label_name]}
        return features,labels

    # 建立读取管道
    def input_pipeline(self, files, type):
        filenames = make_filepath(self.dir, files)
        if type == "train":
            # filenames = self.train_files
            col_names = self.col_name_sorted_with_label
            defaults = self.defaults_with_label
            epoches = self.train_epoches
            labeled = True
            batch_size = self.batch_size
            min_after_dequeue = 10000
            shuffle = True
        elif type == "valid":
            # filenames = self.valid_files
            col_names = self.col_name_sorted_with_label
            defaults = self.defaults_with_label
            epoches = 1
            shuffle = False
            min_after_dequeue = 10000
            labeled = True
            batch_size = self.valid_batch_size
        elif type == "predict":
            # filenames = self.predict_files
            col_names = self.col_name_sorted_without_label
            defaults = self.defaults_without_label
            epoches = 1
            labeled = False
            shuffle = False
            min_after_dequeue = 0
            batch_size = 100


        # 创建文件名队列
        filename_queue = tf.train.string_input_producer( filenames, num_epochs=epoches, shuffle=True )

        record = self.read_my_file_format(filename_queue,col_names,defaults)

        features,labels = self.preprocess(record,labeled)

        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size


        capacity = min_after_dequeue + 3 * batch_size
        if labeled:
            features = dict(features, **labels )
        else:
            features = features

        batched = tf.train.shuffle_batch(
            features,
            batch_size=batch_size,
            capacity=capacity,
            enqueue_many=True,
            min_after_dequeue=min_after_dequeue)


        if labeled:
            label_batch = batched.pop(self.label_name)
            return batched,label_batch
        else:
            return batched


    # 训练和测试
    def train(self, train_files, valid_files, profiling = False):
        #print("model dir: %s" % model_dir)
        monitors = []

        eval_step = int(self.eval_sample_interval / self.batch_size)

        # def input_fn_train():
        #     print('input_fn_train called!')
        #     return self.input_pandas_queue('train')

        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            #input_fn = self.input_pandas_queue('valid'),
            input_fn = lambda :self.input_pipeline(valid_files, 'valid'),
            eval_steps=self.valid_sum/self.valid_batch_size,
            every_n_steps=eval_step
        )

        monitors.append(validation_monitor)

        if profiling:
            print("profiling.......................")
            profile_hook = ProfilerHook(save_secs=3000, output_dir="profiling")
            #profile_hook = tf.train.SessionRunHook()
            monitors.append(profile_hook)

        print("begin to fit....................")

        # 训练模型
        self.model.fit(
            #input_fn=self.input_pandas_queue('train'),
            # x = self.input_pandas_queue_x('train'),
            # y = self.input_pandas_queue_y('train'),
            # batch_size = self.batch_size,
            input_fn= lambda : self.input_pipeline(train_files, 'train'),
            monitors=monitors
        )


        return

    def eval(self,files):
        print("begin to evaluate................")
        self.model.evaluate(input_fn=lambda : self.input_pipeline(files, 'valid'))
        # self.model.evaluate(input_fn=self.input_pandas_queue('valid'))

    def predict(self, files):
        print('begin to predict.................')
        yield self.model.predict(
            input_fn= lambda : self.input_pipeline(files, 'predict')
        )

    def run(self, train_files, valid_files, profiling):
        print("begin to run")
        self.train(train_files, valid_files, profiling)
        # self.eval()
        print("run finished")

if __name__ == "__main__":
    #wdl = WDL('./model/samples','wdl',5,3,[500,500,500],TRAIN_FILELIST,VALID_FILELIST,DES_FILENAME)
    #train_list = ['./data/train_p_e_m_t_c.csv']
    #valid_list = ['./data/valid_p_e_m_t_c.csv']
    #train_list = ['./data/test.csv']
    #valid_list = ['./data/test.csv']
    train_list = ['tf_feature_train_splited.0.csv','tf_feature_train_splited.1.csv','tf_feature_train_splited.2.csv','tf_feature_train_splited.3.csv']
    valid_list = ['tf_feature_train_splited.4.csv']
    des_file = 'feature_des.csv'
    cross_des_file = 'feature_cross_des.csv'
    wdl = WDL('./data','trail16','wdl',300,2,0.1,[500,250,250],des_file,cross_des_file)
    # wdl.run(True)




