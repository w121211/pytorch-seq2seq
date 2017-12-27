def train_options(parser):
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-exp_dir', type=str, default='./experiment')
    parser.add_argument('-load_from', type=str, default=None)

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                            decoded sentences""")

    return parser
