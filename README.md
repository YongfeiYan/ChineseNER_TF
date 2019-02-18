# ChineseNER_TF
Tensorflow implementation of BiLSTM CRF for Chinese NER.

This implementation refers mainly to https://github.com/zjy-ucas/ChineseNER with the following modifications:

- Use personal tfutils to construct model and train.  Fewer codes are needed. 
- The model is tag-schema agnostic. The data format:
  char \t tag
  char \t tag
  ...
- Different evaluation script is used
- Tried different experiments



# Dependencies

    python3.6
    sacred==0.7.4
    tensorflow==1.11.0
    jieba==0.37
    torch==0.4
    torchtext==0.3.1
    seqeval==0.05
To run the code:

```bash
# default configuration
python main.py -F data/exp
# No seg information
python main.py -F data/exp with model.seg_dim=0
# No pretrained embedding
python main.py -F data/exp with emb_path=""
```



## Experiment results

Test results using model with best evalutaion f1 score.

| Settings                    | Precision | Recall | F1     |
| --------------------------- | --------- | ------ | ------ |
| Default                     | 0.9183    | 0.8978 | 0.9076 |
| No pretrained embedding     | 0.9067    | 0.8978 | 0.9021 |
| No word segment information | 0.9053    | 0.8853 | 0.8952 |



# Model

```
Default configufation:
model = {
'char_dim': 100,
'num_chars': 0,
'seg_dim': 20,
'num_segs': 0,
'lstm_dim': 100,
'num_tags': 0,
'dropout_keep': 0.5,
}
emb_path = 'wiki_100.utf8'
optimizer = 'adam'
lr = 0.001
clip_by_value = 5
epochs = 100
data = {
'data_dir': 'data/example',
'batch_size': 20,
'vocab_size': 10000,
}

Model structure:
LSTMCRF(
  (char_inputs): <tf.Tensor 'ChatInputs:0' shape=(?, ?) dtype=int32>
  (char_embedding): Embedding(
    trainable: True
    num_embeddings: 4314
    embedding_dim: 100
    max_norm: None
  )
  (seg_inputs): <tf.Tensor 'SegInputs:0' shape=(?, ?) dtype=int32>
  (seg_embedding): Embedding(
    trainable: True
    num_embeddings: 4
    embedding_dim: 20
    max_norm: None
  )
  (dropout): Dropout(
    keep_prob: 0.5
    (keep_prob): <tf.Tensor 'Dropout_1/keep_prob:0' shape=() dtype=float32>
  )
  (lstm): LSTM(
    num_units: 100
    num_layers: 1
    bidirectional: True
    keep_prob: None
  )
  (lengths): <tf.Tensor 'Sum:0' shape=(?,) dtype=int32>
  (batch_size): <tf.Tensor 'strided_slice:0' shape=() dtype=int32>
  (num_steps): <tf.Tensor 'strided_slice_1:0' shape=() dtype=int32>
  (projection): Sequential(
    Dense(
      Dense_2/weight:0: <tf.Variable 'Dense_2/weight:0' shape=(100, 7) dtype=float32_ref>
      Dense_2/bias:0: <tf.Variable 'Dense_2/bias:0' shape=(7,) dtype=float32_ref>
      <function linear at 0x7f5052eff730>
    )
    Dense(
      Dense_1/weight:0: <tf.Variable 'Dense_1/weight:0' shape=(200, 100) dtype=float32_ref>
      Dense_1/bias:0: <tf.Variable 'Dense_1/bias:0' shape=(100,) dtype=float32_ref>
      <function tanh at 0x7f5052eff598>
    )
  )
  (logits): <tf.Tensor 'add_1:0' shape=(?, ?, 7) dtype=float32>
  (targets): <tf.Tensor 'Targets:0' shape=(?, ?) dtype=int32>
  (loss): <tf.Tensor 'crf_loss/loss:0' shape=() dtype=float32>
)
```

