import argparse

def _add_common_args(parser):
    parser.add_argument("--ckpt-dir", type=str)
    parser.add_argument("--domain", type=str, required=True,
                        choices=['arggen', 'opinion', 'news', 'rst'])
    parser.add_argument("--setup", type=str, required=True,
                        choices=['seq2seq', 'kpseq2seq', 'pair-light', 'pair-full'])
    parser.add_argument("--quiet", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpus", type=int, default=1)

    parser.add_argument("--eval-batch-size", type=int, default=2)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-prompt-len", type=int, default=None,
                        help="Input prompt truncate limit.")
    parser.add_argument("--max-kp-len", type=int, default=None,
                        help="Keyphrase truncate limit.")
    parser.add_argument("--max-tgt-len", type=int, default=None,
                        help="Target side truncate limit, if set to None, default"
                             "values will be used.")


def _add_train_args(parser):
    parser.add_argument("--train-set", required=True, type=str)
    parser.add_argument('--valid-set', required=True, type=str)
    parser.add_argument('--train-batch-size', required=True, type=int)
    parser.add_argument('--num-train-epochs', default=20, type=int)

    parser.add_argument('--warmup-steps', default=1000, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)


    parser.add_argument('--fp16-opt-level', default='O2')

    parser.add_argument('--bart-scratch', action='store_true')

    parser.add_argument('--save-topk-ckpt', default=5, type=int)
    parser.add_argument('--tensorboard-dir', required=True, type=str)
    
    


def _add_generation_args(parser):
    parser.add_argument("--discrim-ckpt-dir", type=str)
    parser.add_argument('--test-set', required=True, type=str)
    parser.add_argument('--chunk-id', type=str)
    parser.add_argument('--output-name', default=None, required=True,
                        help='output name ')

    parser.add_argument('--enforce-template-strategy', default='none', type=str,
                        choices=['none', 'force', 'flexible'],
                        help='Strategy to enforce template (kp) words in output.')

    parser.add_argument('--sample-times', type=int, default=3)
    parser.add_argument('--sampling-topk', default=100, type=int)
    parser.add_argument('--sampling-topp', default=0.99, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--use-system-plan', action='store_true')

    parser.add_argument('--exempt-p', type=float, default=0.5,
                        help='If a token has probability higher than this'
                             'value, it won\'t be masked.')
    parser.add_argument('--low-kp-prob-threshold', type=float, default=0.1,
                        help='When --mask-strategy is set to `non-kp-worst-k`, use'
                             'this number to mask KP that has lower probability'
                        )
    parser.add_argument('--do-sampling', action='store_true')
    parser.add_argument('--iterations', default=5, type=int)
    
    parser.add_argument('--use-pplm', action='store_true')
    parser.add_argument('--do-test', action='store_true')
    parser.add_argument('--no-rst', action='store_true')
    parser.add_argument('--no-bow', action='store_true')
    

def _add_pplm_args(parser):
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.0001)
    #parser.add_argument("--temperature", type=float, default=1.0)
    #parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.5)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    #parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")

def get_train_parser():
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    _add_train_args(parser)
    return parser

def get_generation_parser():
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    _add_generation_args(parser)
    _add_pplm_args(parser)
    return parser