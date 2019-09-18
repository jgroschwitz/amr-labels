// jsonnet allows local variables like this
local embedding_dim = 64;
local hidden_dim = 64;
local num_epochs = 5;
local patience = 5;
local batch_size = 64;
local learning_rate = 0.001;

{
    "train_data_path": 'data_formatting/amr17/dev/gold-dev.amconll',
    "validation_data_path": 'data_formatting/amr17/dev/gold-dev.amconll',
    "dataset_reader": {
        "type": "amconll-aligned-lex"
    },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            // Technically you could put a "type": "basic" here,
            // but that's the default TextFieldEmbedder, so doing so
            // is optional.
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "num_layers": 2, #TWO LAYERS, we don't use sesame street.
            "recurrent_dropout_probability": 0.4,
            "layer_dropout_probability": 0.3,
            "input_size": embedding_dim,
            "use_highway": false,
            "hidden_size": hidden_dim,
        }
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
