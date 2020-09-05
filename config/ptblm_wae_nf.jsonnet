local SEED = 0;
local TOKENIZER = {
  "type": "spacy",
  "split_on_spaces": true
};
local READER = {
  "type": "autoencoder",
  "tokenizer": TOKENIZER,
};
local CUDA = 0;

local EMBEDDING_DIM = 200;
local HIDDEN_DIM = 200;
local LATENT_DIM = 32;
local BATCH_SIZE = 128;
local NUM_LAYERS = 1;
local BIDIRECTIONAL = true;

local NUM_EPOCHS = 48;
local PATIENCE = 45;
local SUMMARY_INTERVAL = 10;
local GRAD_NORM = 5.0;
local SHOULD_LOG_PARAMETER_STATISTICS = false;
local SHOULD_LOG_LEARNING_RATE = true;
local OPTIMIZER = "adam";
local LEARNING_RATE = 0.001;
local INIT_UNIFORM_RANGE_AROUND_ZERO = 0.1;

local MMD_WEIGHT = {
  "type": "constant_weight",
  "initial_weight": 10.0,
};

local KL_WEIGHT = {
  "type": "linear_annealed",
  "slope": 1/6580,
  "intercept": -0.15,
  "max_weight": 0.8
};

{
  "random_seed": SEED,
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "dataset_reader": READER,
  "train_data_path": "data/raw/ptb/train.txt",
  "validation_data_path": "data/raw/ptb/valid.txt",
  "model": {
    "type": "lvm",
    "variational_encoder": {
      "type": "gaussian",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "tokens",
            "embedding_dim": EMBEDDING_DIM,
          }
        }
      },
      "encoder": {
        "type": "lstm",
        "input_size": EMBEDDING_DIM,
        "hidden_size": HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        "bidirectional": BIDIRECTIONAL,
      },
      "latent_dim": LATENT_DIM,
    },
    "variational_loss": {
      "kl_weight": KL_WEIGHT,
      "mmd_weight": MMD_WEIGHT,
    },
    "decoder": {
      "type": "variational_decoder",
      "target_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "tokens",
            "embedding_dim": EMBEDDING_DIM,
          }
        }
      },
      "rnn": {
        "type": "lstm",
        "input_size": EMBEDDING_DIM + LATENT_DIM,
        'num_layers': NUM_LAYERS,
        "hidden_size": HIDDEN_DIM,
      },
      "latent_dim": LATENT_DIM,
      "dropout_p": 0.2
    },
    "flow": {
      "type": "planar",
      "num_flows": 3,
      "input_dim": LATENT_DIM
    },
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": BATCH_SIZE,
    }
  },
  "trainer": {
    "type": 'gradient_descent',
    "num_epochs": NUM_EPOCHS,
    "cuda_device": CUDA,
    "optimizer": {
      "type": OPTIMIZER,
      "lr": LEARNING_RATE
    },
    "epoch_callbacks": ["print_reconstruction_example", "print_generation_example"],
    "patience": PATIENCE,
    "validation_metric": "-nll"
  }
}
