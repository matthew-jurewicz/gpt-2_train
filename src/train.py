import os, sys
import numpy as np
from model import *
from encoder import *
import tensorflow as tf

def split_data(data_dir, test_split=.2):
    files = os.listdir(data_dir)

    train_dir = os.path.join(data_dir, 'train')
    os.mkdir(train_dir)
    test_dir = os.path.join(data_dir, 'test')
    os.mkdir(test_dir)

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            text = f.read()
            i = int(len(text) * test_split)
            
        with open(os.path.join(train_dir, filename), 'w+') as f:
            f.write(text[:-i])

        with open(os.path.join(test_dir, filename), 'w+') as f:
            f.write(text[i:])

def get_batch(data, batch_size, seq_len, vocab_size):
    for i in range(len(data) // seq_len):
        idxs = np.random.choice(a=len(data) - seq_len - 1, 
                                size=batch_size, 
                                replace=False)
        tmp = np.array([data[j:j + seq_len + 1] for j in idxs])
        x = tmp[:,:-1]
        y = tf.keras.utils.to_categorical(tmp[:,1:], num_classes=vocab_size)

        yield x, y

if __name__ == '__main__':
    root_dir = os.path.abspath(os.pardir)
    models_dir = os.path.join(root_dir, 'models')
    model_name = '345M'
    hparams_filepath = os.path.join(models_dir, model_name, 'hparams.json')
    hparams = default_hparams()
    with open(hparams_filepath, 'r') as f:
        hparams.parse_json(f.read())

    with tf.get_default_graph().as_default():
        tpu = 'grpc://ip.address.of.tpu:8470'
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)

        with tpu_strategy.scope():
            batch_size = 32#NOTE: reduce batch size if OOM
            seq_len = hparams.n_embd
            vocab_size = hparams.n_vocab
            inputs = tf.placeholder(tf.int32, (None, seq_len))
            logits = model(hparams, inputs)['logits']

            learn_rate = 6.25e-5
            drop_rate = .1
            labels = tf.placeholder(tf.int32, (None, seq_len, vocab_size))
            xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
            train_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(xentropy)

        def load_data(data_dir, bpe):
            text = ''
            for filename in os.listdir(data_dir):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    text += f.read()
                    text += '\r\n'
            data = bpe.encode(text)

            return data

        data_dir = os.path.join(root_dir, 'data')
        bpe = get_encoder(model_name, models_dir)
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        train_data = load_data(train_dir, bpe)
        test_data = load_data(test_dir, bpe)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=tf.trainable_variables(scope='model'))
        ckpt_filepath = os.path.join(models_dir, model_name, 'model.ckpt')
        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, ckpt_filepath)

            nepochs = 100
            for i in range(nepochs):
                for x, y in get_batch(train_data, batch_size, seq_len, vocab_size):
                    _, loss = sess.run([train_op, xentropy], feed_dict={inputs:x, labels:y, 'drop_rate:0':drop_rate})

                val_loss = 0
                count = 0
                for x, y in get_batch(test_data, batch_size, seq_len, vocab_size):
                    val_loss += sess.run([xentropy], feed_dict={inputs:x, labels:y})[0]
                    count += 1
                val_loss /= count

                print('epoch {}: loss -> {:.4f}\t\tval_loss -> {:.4f}'.format(i + 1, loss, val_loss))
                save_path = saver.save(sess, ckpt_filepath, global_step=i)
                print(save_path)
                sys.stdout.flush()