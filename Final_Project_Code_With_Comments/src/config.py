import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for COLING Shared Task 2022")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    
    # ========================= Data Configs ==========================
    parser.add_argument('--train_path', type=str, default='data/train.tsv')
    parser.add_argument('--valid_path', type=str, default='data/validation.tsv')
    parser.add_argument('--test_path', type=str, default='data/validation.tsv')
    
    
    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='data/checkpoint/')
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--best_score', default=0.75, type=float, help='save checkpoint if mean_f1 > best_score')
    
    # ========================= Learning Configs ==========================
    parser.add_argument('--batch_size', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=64, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=64, type=int, help="use for testing duration per worker")
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=100, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1500, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--warmup_rate', default=0.1, type=float, help="warm rate for parameters not in bert or vit")
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help="gradient accumulation steps")
    
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    
    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='roberta-base')
    parser.add_argument('--bert_cache', type=str, default='data/bert_cache/')
    parser.add_argument('--bert_embedding_size', type=int, default=768, help='bert embedding size')
    parser.add_argument('--bert_hidden_size', type=int, default=768, help='bert output size')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    parser.add_argument('--bert_learning_rate', type=float, default=2e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
    # ===========================GPU ID ===========================
    parser.add_argument('--gpu_ids', type=str, default='0')
    

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # ===========================Mix up =========================
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0)
    
    # ===========================Adversarial training =========================
    parser.add_argument('--fgm', type=int, default=0, help="A value greater than 0 indicates use, and 0 indicates no use")
    parser.add_argument('--pgd', type=int, default=0, help="A value greater than 0 indicates use, and 0 indicates no use")
    
    # ===========================rdrop =========================
    parser.add_argument('--use_rdrop', action='store_true', help='Using this parameter indicates that R-drop is used')
    
    # ===========================Model verification =========================
    parser.add_argument('--save_steps', type=int, default=100, help="save model")
    # ===========================Whether to use multi-card training =========================
    parser.add_argument('--distributed_train', action='store_true', help='The training was performed using Doka')
    # ===========================Task label category=========================
    parser.add_argument('--class_label', type=int, default=3, help="Number of classes")
    # ===========================Does not change whether learning rate is used or not =========================
    parser.add_argument('--constant_lr', action='store_true', help='whether to use get_constant_schedule_with_warmup')
    
    # ===========================SWA =========================
    parser.add_argument('--swa_start', type=int, default=-1, help="SWA start epoch")
    parser.add_argument('--swa_lr', type=float, default=5e-5)
    # ===========================EMA =========================
    parser.add_argument('--ema_start', type=int, default=-1, help="EMA start step")
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    # ====================================================
    parser.add_argument('--premise', action='store_true', help='Whether to perform premise classification')
    
    # ====================================================
    parser.add_argument('--result_file', type=str, default='result.tsv')
    # ===========================focal loss =========================
    parser.add_argument('--gamma_focal', type=float, default=-1,help="A value greater than 0 indicates the use of focal loss")
    
    # =========================== Whether to use a continuously pretrained model for fine-tuning=========================
    parser.add_argument('--pretrain_model_path', type=str, default=None)

    # =========================== Whether to freeze training embeddings=========================
    parser.add_argument('--embedding_freeze', action='store_true', help='Whether to freeze embeddings')

    
    return parser.parse_args()