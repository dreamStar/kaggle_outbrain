# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import math
from profile_hook import ProfilerHook
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)

HIDDEN_UNITS = [1024,512,256]
TRAIN_FILELIST = ["data_sample.csv"]
VALID_FILELIST = ["data_sample.csv"]
DES_FILENAME = "data_sample_des.csv"

class WDL(object):
    def __init__(self,model_dir,model_type,batch_size,train_epoches,dnn_lr,hidden_units,train_files,valid_files,des_file):
        self.model_dir = model_dir
        self.model_type = model_type
        self.batch_size = batch_size
        self.train_epoches = train_epoches
        self.train_files = train_files
        self.valid_files = valid_files
        self.des_file = des_file
        self.hidden_units = hidden_units
        self.eval_sample_interval = 1000000
        self.save_ckpt_interval = 1000000
        self.save_summary_interval = 1000000
        self.dnn_lr = dnn_lr
        self.write_params()


        self.read_col_des(self.des_file)
        self.get_cols()
        self.build_estimator()




    def write_params(self):
        params = {
            "modle_type" : self.model_type,
            "batch_size" : self.batch_size,
            "train_epoches" : self.train_epoches,
            "train_files" : self.train_files,
            "valid_files" : self.valid_files,
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
    def read_col_des(self,filename):
        self.col_name_all = []
        self.col_name_all_without_label = []
        self.col_name_in_use = []
        self.col_val_default = []
        self.col_val_default_without_label = []
        self.col_des = {}
        self.label_name = None
        with open(filename,'r') as f:
            for line in f:
                print("des line:")
                array = line.strip().split("\t")
                print("array:")
                print(array)
                self.col_name_all.append(array[0])

                if array[1] == "label":
                    self.label_name = array[0]
                    self.col_val_default.append([0])
                    continue
                else:
                    self.col_name_all_without_label.append(array[0])

                if array[2] != "true":
                    self.col_des[array[0]] = {
                        "type" : array[1],
                        "enable" : False
                    }
                    self.col_val_default.append(0)
                    self.col_val_default_without_label.append(0)
                    continue

                self.col_des[array[0]] = {
                    "type" : array[1],
                    "enable" : array[2] == "true",
                    "model" : array[3],
                    "method" : array[5],
                    "param" : array[6],
                    "embedding_size" : int(array[7])
                }



                if array[1] == "int":
                    self.col_val_default.append([int(array[4])])
                    self.col_val_default_without_label.append(tf.constant(int(array[4])))
                    self.col_des[array[0]]["default"] = int(array[4])
                elif array[1] == "float":
                    self.col_val_default.append([float(array[4])])
                    self.col_val_default_without_label.append(tf.constant(float(array[4])))
                    self.col_des[array[0]]["default"] = float(array[4])
                else:
                    self.col_val_default.append([array[4]])
                    self.col_val_default_without_label.append(tf.constant(array[4]))
                    self.col_des[array[0]]["default"] = array[4]

                if self.col_des[array[0]]["enable"] :
                    self.col_name_in_use.append(array[0])

    # 构建列
    def get_cols(self):
        layers = tf.contrib.layers
        self.wide_cols = []
        self.deep_cols = []

        # 仅处理部分属性
        for name in self.col_des:
            col_attr = self.col_des[name]
            if not col_attr["enable"]:
                continue
            if col_attr["type"] == "label":
                continue

            if col_attr["model"] == "wide" or col_attr["model"] == "wdl" :

                if col_attr["method"] == "indexed":
                    self.wide_cols.append(layers.sparse_column_with_integerized_feature(column_name=name,bucket_size=int(col_attr["param"])))
                elif col_attr["method"] == "feed":
                    self.wide_cols.append(layers.real_valued_column(column_name=name, default_value=col_attr['default']))

            if col_attr["model"] == "deep" or col_attr["model"] == "wdl" :

                if col_attr["method"] == "indexed":
                    self.deep_cols.append(layers.embedding_column(layers.sparse_column_with_integerized_feature(column_name=name,bucket_size=int(col_attr["param"])),dimension=col_attr["embedding_size"]))
                elif col_attr["method"] == "feed":
                    self.wide_cols.append(layers.real_valued_column(column_name=name, default_value=col_attr['default']))


        # 建立预测器
    def build_estimator(self):
        """Build an estimator."""
        save_ckpt = int(self.save_ckpt_interval / self.batch_size)
        save_summary = int(self.save_summary_interval / self.batch_size)

        if self.model_type == "wide":
            self.model = tf.contrib.learn.LinearClassifier(
                model_dir=self.model_dir,
                feature_columns=self.wide_cols,
                config=tf.contrib.learn.RunConfig(gpu_memory_fraction = 0.6, log_device_placement=True,save_summary_steps=save_summary, save_checkpoints_steps=save_ckpt,save_checkpoints_secs=None)
            )
        elif self.model_type == "deep":
            self.model = tf.contrib.learn.DNNClassifier(
                model_dir=self.model_dir,
                feature_columns=self.deep_cols,
                hidden_units=self.hidden_units,
                optimizer=tf.train.AdagradOptimizer(self.dnn_lr),
                config=tf.contrib.learn.RunConfig(gpu_memory_fraction = 0.6, log_device_placement=True,save_summary_steps=save_summary, save_checkpoints_steps=save_ckpt,save_checkpoints_secs=None)
            )
        else:
            self.model = tf.contrib.learn.DNNLinearCombinedClassifier(
                model_dir=self.model_dir,
                linear_feature_columns=self.wide_cols,
                dnn_feature_columns=self.deep_cols,
                dnn_hidden_units=self.hidden_units,
                dnn_optimizer=tf.train.AdagradOptimizer(self.dnn_lr),
                config=tf.contrib.learn.RunConfig(gpu_memory_fraction = 0.6, log_device_placement=True,save_summary_steps=save_summary, save_checkpoints_steps=save_ckpt,save_checkpoints_secs=None)
            )
        return


    # 数据读取操作符
    def read_my_file_format(self,filename_queue,col_names, defaults):
        reader = tf.TextLineReader()
        key, record_string = reader.read(filename_queue)

        cols = tf.decode_csv(record_string, record_defaults = defaults )

        #cols = tf.decode_csv(record_string )
        record = dict([(col_names[i],cols[i]) for i in xrange(len(cols))])

        return record


    # 数据预处理
    def preprocess(self, record, labeled):
        if not labeled :
            features = record
            return features

        features = dict([(self.col_name_in_use[i], record[self.col_name_in_use[i]]) for i in xrange(len(self.col_name_in_use)) if self.col_name_in_use[i] != self.label_name])
        labels = {self.label_name:record[self.label_name]}
        return features,labels

    # 建立读取管道
    def input_pipeline(self,type):
        if type == "train":
            filenames = self.train_files
            col_names = self.col_name_all
            defaults = self.col_val_default
            epoches = self.train_epoches
            labeled = True
            min_after_dequeue = 10000
            batch_size = self.batch_size
        elif type == "valid":
            filenames = self.valid_files
            col_names = self.col_name_all
            defaults = self.col_val_default
            epoches = 1
            labeled = True
            batch_size = 10000
            min_after_dequeue = 0
        elif type == "predict":
            filenames = self.predict_files
            col_names = self.col_name_all_without_label
            defaults = self.col_val_default_without_label
            epoches = 1
            labeled = False
            min_after_dequeue = 0
            batch_size = 100000


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
            features, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)


        if labeled:
            label_batch = batched.pop(self.label_name)
            return batched,label_batch
        else:
            return batched


    def input_pandas_queue(self,type):
        if type == "train":
            filenames = self.train_files
            col_names = self.col_name_all
            defaults = self.col_val_default
            epoches = self.train_epoches
            labeled = True
            min_after_dequeue = 10000
            batch_size = self.batch_size
        elif type == "valid":
            filenames = self.valid_files
            col_names = self.col_name_all
            defaults = self.col_val_default
            epoches = 1
            labeled = True
            batch_size = 10000
            min_after_dequeue = 0
        elif type == "predict":
            filenames = self.predict_files
            col_names = self.col_name_all_without_label
            defaults = self.col_val_default_without_label
            epoches = 1
            labeled = False
            min_after_dequeue = 0
            batch_size = 100000

        data = None
        cols = col_names
        for f in filenames:
            this_data = pd.read_csv(f, sep=',', header=None, names=cols)
            print('this data: %i' % this_data.size)
            if data :
                data = pd.concat([data, this_data])
            else:
                data = this_data
        print('data size: %i' % data.size)

        # data = {k: tf.constant(data[k].values) for k in col_names}
        # feature_cols_batch = tf.train.shuffle_batch(data, batch_size=batch_size,capacity=1000,
        #     min_after_dequeue=0, enqueue_many=True, allow_smaller_final_batch=True)
        #
        # if labeled:
        #     label_batch = feature_cols_batch.pop(self.label_name)
        #
        # return feature_cols_batch, label_batch

        steps = int(math.floor(data.size + batch_size - 1 / batch_size))
        for i in xrange(epoches):
            print('epoche %i' % i)
            data = data.reindex(np.random.permutation(data.index))
            for step in xrange(steps):
                indexes = range(batch_size*step, min(data.size, batch_size * (step + 1)))
                data_slice = data.iloc[indexes]
                yield {k: tf.constant(data_slice[k].values) for k in self.col_name_all_without_label}, tf.constant(data_slice[self.label_name].astype('int'))



    # 训练和测试
    def train(self,profiling = False):
        #print("model dir: %s" % model_dir)
        monitors = []

        eval_step = int(self.eval_sample_interval / self.batch_size)

        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn = lambda :self.input_pipeline('valid'),
            eval_steps=50,
            every_n_steps=eval_step
        )

        monitors.append(validation_monitor)

        if profiling:
            print("profiling.......................")
            profile_hook = ProfilerHook(save_secs=30, output_dir="profiling")
            #profile_hook = tf.train.SessionRunHook()
            monitors.append(profile_hook)

        print("begin to fit....................")

        # 训练模型
        self.model.fit(
            #input_fn=lambda: self.input_pipeline('train'),
            input_fn=lambda: self.input_pandas_queue('train'),
            monitors=monitors
        )


        return

    def eval(self):
        print("begin to evaluate................")
        self.model.evaluate(input_fn=lambda : self.input_pipeline('valid'))



    def run(self,profiling):
        print("begin to run")
        self.train(profiling)
        self.eval()
        print("run finished")

if __name__ == "__main__":
    #wdl = WDL('./model/samples','wdl',5,3,[500,500,500],TRAIN_FILELIST,VALID_FILELIST,DES_FILENAME)
    train_list = ['./data/train_p_e_m_t_c.csv']
    valid_list = ['./data/valid_p_e_m_t_c.csv']
    #train_list = ['./data/test.csv']
    #valid_list = ['./data/test.csv']
    wdl = WDL('./data/trail15','deep',100,2,0.1,[500,250,250],train_list,valid_list,'./data/click_event_promoted_topic_category_des.csv')
    wdl.run(True)



