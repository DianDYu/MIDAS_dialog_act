# dialog_act_bert

### Dataset preparation
Example Input format in the including file folder ./da_data: 

chatbot utterance : previous user utterance > current user utterance ## dialog act 1 of current user utterance;dialog act 2 of current user utterance

EMPTY means that there is no previous user utterance and the current user utterance is the first utterance responding to the chatbot

For example the dialog below with user current utterance's dialog act annotated as pos_answer can be formatted as: "did you know that : yes > i did ## pos_answer;"

Chatbot: did you know that?

User: yes

User: i did(dialog act: pos_answer)

Another example dialog shown below can be formatted as : "do you want to hear some fun facts about cats instead : EMPTY > yes ## pos_answer;command". In this case, there is only current user utterance so the previous utterance is input as EMPTY

Chatbot: do you want to hear some fun facts about cats instead

User: yes(dialog act: pos_answer and command)

### Train and evaluate the act prediction model
```
python run_classifier.py --data_dir da_data/ --bert_model bert-base-uncased --task_name da --output_dir output --do_eval --binary_pred  --do_train
```
* --data_dir: the data directory where training and evaluating data file is stored. By default, the training data file is named as 'train.txt', and the evaluating data file is named as 'dev.txt'
* --bert_model: bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese
* --task_name: the name of the task to train, for example da represents dialog act 
* --output_dir: the output directory where the model predictions and checkpoints will be written
* --do_eval: whether to evaluate on the evaluation file
* --do_train: whether to run training
* --binary_pred: whether to use Binary-Cross-Entropy for binary prediction instead of only one tag(in case of multi-dialog-act
