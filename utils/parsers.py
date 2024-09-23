import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-preprocess', type=bool, default=False, help="preprocess dataset")    #when you first run code, you should set it to true.

##### model parameters
parser.add_argument('-data_name', type=str, default='dblp', choices=['weibo', 'meme',  'dblp'], help="dataset")
# parser.add_argument('-data_name', type=str, default='meme', choices=['weibo', 'meme',  'dblp'], help="dataset")
# parser.add_argument('-data_name', type=str, default='christianity', choices=['weibo', 'meme',  'dblp'], help="dataset")
# parser.add_argument('-data_name', type=str, default='android', choices=['weibo', 'meme',  'dblp'], help="dataset")
parser.add_argument('-epoch', type=int, default=200)

parser.add_argument('-max_lenth', type=int, default=100) #200=========================================
parser.add_argument('-batch_size', type=int, default=64) #64
parser.add_argument('-posSize', type=int, default=8, help= "the position embedding size")
parser.add_argument('--embSize', type=int, default=128, help='emb edding size')#64
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')#1e-5
parser.add_argument('--lr', type=float, default=0.001, help='learning rate') #0.001
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')#3
parser.add_argument('--beta', type=float, default= 0.002, help='ssl graph task maginitude')  #0.001
parser.add_argument('--window', type=int, default=10, help='window size')  #10=========================
parser.add_argument('-n_warmup_steps', type=int, default=1000) #1000
parser.add_argument('-dropout', type=float, default=0.25) #0.2 
parser.add_argument('-use_doc', type=bool, default=True, help='use text') 
# parser.add_argument('-use_doc', type=bool, default=False, help='use text') 
parser.add_argument('--fusion-method', '-f', choices=['mean', 'max', 'cat', 'sum'], default='cat',
                    help='The fusion method to use for combining the hidden and dy_emb.')


#####data process
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.2)

###save model
parser.add_argument('-save_path', default= "./checkpoint/")
parser.add_argument('-patience', type=int, default=9, help="control the step of early-stopping")