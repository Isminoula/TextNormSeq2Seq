# TextNormalization with Seq2Seq 

This is code repo for our AAAI ICWSM 2019 paper "Adapting Sequence to Sequence models for Text Normalization in Social Media", 
where we explore several Seq2Seq variations for normalizing social media text.

If you find this code, models or results useful, please cite us using the following bibTex:
```
@inproceedings{lourentzou2019adapting,
  title={Adapting Sequence to Sequence models for Text Normalization in Social Media},
  author={Lourentzou, Ismini and Manghnani, Kabir and Zhai, ChengXiang},
  booktitle={International Conference on Web and Social Media},
  year={2019},
  organization={AAAI}
}
```

### Requirements
- torch==0.4.1
- python 2.7


### Download the Lexnorm2015 dataset
```bash
mkdir dataset
cd dataset
wget https://github.com/noisy-text/noisy-text.github.io/raw/master/2015/files/lexnorm2015.tgz
tar -zxvf lexnorm2015.tgz
cp lexnorm2015/* .
rm -rf lexnorm2015 lexnorm2015.tgz
cd ..
```

### Training a hybrid Seq2Seq model from scratch 
The hybrid model is a combination of two Seq2Seq models: a word-level one (**S2S**) 
and a secondary character-level trained on pairs of words (spelling with noise augmented data).

i) Train a word-level model, save results in folder `word_model` 
```bash
python main.py -logfolder -save_dir word_model -gpu 0 -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.5 -emb_size 100 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 5 -rnn_size 200 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```
ii) Train a secondary character-level model, save results in folder `spelling_model`
```bash
python main.py -logfolder -save_dir spelling_model -gpu 0 -input spelling -data_augm -noise_ratio 0.1 -attention -bias -lowercase -bos -eos -brnn -batch_size 500 -dropout 0.5 -emb_size 256 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.001 -max_grad_norm 5 -rnn_size 500 -rnn_type 'LSTM'  -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 30 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```


### Test hybrid Seq2Seq model
Evaluate final model (**HS2S**) combining the trained models described above:
```bash
python main.py -eval -logfolder -save_dir hybrid_model -gpu 0 -load_from word_model/model_50_word.pt -char_model spelling_model/model_50_spelling.pt -input hybrid -data_augm -noise_ratio 0.1 -lowercase -bos -eos -batch_size 32 -share_vocab
```

### Other models 
You can also try some additional models, for example:

- Word-level model where tokens that need no normalization are mapped to a common *@self* token (**S2SSelf**)
```bash
python main.py -logfolder -save_dir S2SSelf -gpu 3 -self_tok -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.2 -emb_size 100 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 10 -rnn_size 100 -rnn_type 'LSTM'  -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```
- Word-level model for non-unique mappings only (**S2SMulti**)
```bash
python main.py -logfolder -save_dir S2SMulti -gpu 4 -correct_unique_mappings -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.5 -emb_size 100 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 5 -rnn_size 200 -rnn_type 'LSTM'  -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```
- Character-level model (**S2SChar**)
```bash
python main.py -logfolder -save_dir S2SChar -gpu 5 -input char -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.2 -emb_size 256 -end_epoch 50 -layers 3 -learning_rate_decay 0.5 -lr 0.001 -max_grad_norm 10 -rnn_size 512 -rnn_type 'LSTM'  -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 30 -teacher_forcing_ratio 0.6  -max_train_decode_len 200
```

 
### Pretrained models - Reproducibility
We have done our best to ensure reproducibility of our results, however this is not always [guaranteed](https://pytorch.org/docs/stable/notes/randomness.html).
As an extra reproducibility step, we also release our best performing models. Just unzip the `pretrained_models.zip` found [here](https://app.box.com/s/8pcntfotwxcytddzximu6goxi1rz92p2) and try them by setting the flag `eval` and updating the model folders in your python commands, for example:
- Pre-trained hybrid model (**HS2S**):
```bash
python main.py -eval -logfolder -save_dir hybrid_model -gpu 0 -load_from pretrained_models/word_model/model_50_word.pt -char_model pretrained_models/spelling_model/model_50_spelling.pt -input hybrid -data_augm -noise_ratio 0.1 -lowercase -bos -eos -batch_size 32 -share_vocab
``` 
- Pre-trained word-level model (**S2S**):
```bash
python main.py -eval -logfolder -gpu 0 -load_from pretrained_models/word_model/model_50_word.pt  -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.5 -emb_size 100 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 5 -rnn_size 200 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```
A separate file `pretrained_models/run.sh` contains all commands for reproducing the aforementioned models.

### Interactive mode
With the `interactive` flag, you can also try the model on any arbitrary text through command line, for example:
```console
foo@bar:~$ python main.py -interactive -gpu 0 -load_from pretrained_models/word_model/model_50_word.pt -char_model pretrained_models/spelling_model/model_50_spelling.pt -input hybrid -data_augm -noise_ratio 0.1 -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab
Please enter the text to be normalized (q to quit): lol how are u doin
Prediction is:laughing out loud how are you doing
Please enter the text to be normalized (q to quit): q 
foo@bar:~$
```

#### Notes
If you wish to work on CPU, simply remove the flag `-gpu 0` from the following commands.

Each command prints in a file named `output.log` saved in defined by `save_dir`. 
Remove the flag `-logfolder` to output to console. 


*Credits to [khanhptnk](https://github.com/khanhptnk/bandit-nmt) and
[howardyclo](https://github.com/howardyclo/pytorch-seq2seq-example) as this project borrows code from their repositories.*

*Also, checkout [MoNoise](https://bitbucket.org/robvanderg/monoise) for a non-DL SoTA model.*

