// jsonnet allows local variables like this
local embedding_dim = 16;
local hidden_dim = 16;
local num_epochs = 300;
local patience = 10000;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": 'data_formatting/amr17/train/toy.amconll',
    "validation_data_path": 'data_formatting/amr17/train/toy.amconll',
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
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        },
        "loss": "reinforce"
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
        "dev_acc_by_bucket_file": 'bucket_accs_toy.csv',
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "adam",
            "lr": learning_rate
        },
        "patience": patience,
        "validation_metric": "+fscore"
    }
}
