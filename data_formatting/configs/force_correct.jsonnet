// jsonnet allows local variables like this
local embedding_dim = 1024; // BERT output dim
local hidden_dim = 256;
local num_epochs = 30;
local patience = 30;
local batch_size = 256;
local learning_rate = 0.004;
local bert_model = "bert-large-uncased";

{
    "train_data_path": 'data_formatting/amr17/train/train.amconll',
    "validation_data_path": 'data_formatting/amr17/dev/gold-dev.amconll',
    "dataset_reader": {
        "type": "amconll-aligned-lex",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
                "do_lowercase": false,
                "use_starting_offsets": true
            }
        }
    },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"],
                "token_characters": ["token_characters"],
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_model
                }
            }
        },
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "num_layers": 2,
            "recurrent_dropout_probability": 0.4,
            "layer_dropout_probability": 0.3,
            "input_size": embedding_dim,
            "use_highway": false,
            "hidden_size": hidden_dim,
        },
        "loss": "force_correct"
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["sentence", "num_tokens"]]
    },
    "trainer": {
        "type": 'jonas',
        "dataset_reader": {
            "type": "amconll-aligned-lex"
        },
        "prediction_log_file": 'prediction_test.log',
        "dev_acc_by_bucket_file": 'acc_by_bucket.csv',
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "adam",
            "lr": learning_rate
        },
        "patience": patience,
        "validation_metric": "+fscore"
    }
}
