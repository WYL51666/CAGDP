import os
import logging
import datetime
import Constants
from tqdm import tqdm
# from models.model_noglu import *
from models.models import *
# from models.model_glu_single import *     
# from models.model_lstm import *
# from pretrain.textemb import *
from torch.utils.data import DataLoader
from dataLoader import datasets, Read_data, Split_data
from utils.parsers import parser
from utils.Metrics import Metrics
from utils.EarlyStopping import *
from utils.graphConstruct import ConHypergraph
metric = Metrics()
opt = parser.parse_args()
def setup_logger(dataname):
    if not os.path.exists(f'Log/{dataname}/'):
        os.makedirs(f'Log/{dataname}/')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    
    file_handler = logging.FileHandler(f'Log/{dataname}/train_{now_str}.log')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger,file_handler

def init_seeds(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data) 
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct

def model_training(model, train_loader, epoch,beta):
    ''' model training '''
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    logger.info('start training: ' + str(datetime.datetime.now()))
    print('start training: ', datetime.datetime.now())
    # training
    model.train() 
    with tqdm(total=len(train_loader),mininterval=2.0) as t:
        for step, (cascade_item, label, cascade_time, label_time, cascade_len) in enumerate(train_loader):

            n_words = label.data.ne(Constants.PAD).sum().float().item()
            n_total_words += n_words

            model.zero_grad()
            cascade_item = trans_to_cuda(cascade_item.long())
            tar = trans_to_cuda(label.long())##label
            # cascade_time = trans_to_cuda(cascade_time.long())
            # label_time = trans_to_cuda(label_time.long())
            pred, ssl_loss = model(cascade_item)#对应LSTMGNN的def forward(self, input, label):
            # pred, ssl_loss, ss_loss2 = model(cascade_item, tar)#对应LSTMGNN的def forward(self, input, label):
            loss, n_correct = get_performance(model.loss_function, pred, tar)
            # loss = loss + 0.0015 * ssl_loss + model.beta2 * ss_loss2
            loss = loss + beta * ssl_loss #+ beta2 * ss_loss2
            loss.backward()#retain_graph=True
            model.optimizer.step()
            model.optimizer.update_learning_rate() 
            
            ### tqdm parameter
            t.set_description(desc="Epoch %i" % epoch)
            t.set_postfix(steps=step, loss=loss.data.item())
            t.update(1)
            total_loss += loss.item()
            n_total_correct += n_correct
        logger.info('\n[Epoch]%s', epoch)
        logger.info('\tTotal Loss:\t%.3f' % total_loss)
        print('\tTotal Loss:\t%.3f' % (total_loss/n_total_words))

        return total_loss/n_total_words, n_total_correct/n_total_words

def model_testing(model, test_loader, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0


    n_total_words = 0.0
    n_correct = 0.0

    logger.info('start predicting: ' + str(datetime.datetime.now()))
    print('start predicting: ', datetime.datetime.now())   
    model.eval()

    with torch.no_grad():
        for step, (cascade_item, label, cascade_time, label_time, cascade_len) in enumerate(test_loader):
        # for i, batch in enumerate(validation_data):  #tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):

            cascade_item = trans_to_cuda(cascade_item.long())
            # cascade_time = trans_to_cuda(cascade_time.long())
            # y_pred = model.model_prediction(cascade_item)
            y_pred, ssl_loss = model(cascade_item)
            y_pred = y_pred.detach().cpu()
            tar = label.view(-1).detach().cpu()

            pred = y_pred.max(1)[1]
            gold = tar.contiguous().view(-1)
            correct = pred.data.eq(gold.data)
            n_correct = correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

            scores_batch, scores_len = metric.compute_metric(y_pred, tar, k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words
            
        return scores, n_correct/n_total_words  

def train_test(epoch, model, train_loader, val_loader, test_loader,beta):

    total_loss, accuracy = model_training(model, train_loader, epoch,beta)
    val_scores, val_accuracy = model_testing(model, val_loader)
    test_scores, test_accuracy = model_testing(model, test_loader)

    return total_loss, val_scores, test_scores, val_accuracy.item(), test_accuracy.item()

def main(data_path, seed=2024):
    
    init_seeds(seed)

    # ========= Preparing DataLoader =========#
    #### Divide training set, validation set and test set
    if opt.preprocess:
        Split_data(data_path, train_rate=0.8, valid_rate=0.1, load_dict=False)
        print('preprocessing')


    #### Read training set, validation set and test set
    train, valid, test, user_size = Read_data(data_path)

    train_data = datasets(train, opt.max_lenth)
    val_data = datasets(valid, opt.max_lenth)
    test_data = datasets(test, opt.max_lenth)

    #### Build DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=6)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=6)

    # ========= Preparing graph and hypergraph =========#
    opt.n_node = user_size
    if opt.use_doc:
        text_embedding = torch.from_numpy(np.load('data/'+data_path+'/text_embs.npy'))
    else:
        text_embedding = None
    # text_embedding = torch.from_numpy(np.load('data/' +data_path+'/text_embs.npy'))
    HG_G, HG_L = ConHypergraph(data_path, opt.n_node, opt.window, opt.use_doc,text_embedding)
    HG_G = trans_to_cuda(HG_G)
    HG_L = trans_to_cuda(HG_L)
    if opt.use_doc:
        text_embedding = trans_to_cuda(text_embedding)
    # 定义参数网格
    # param_grid = {
    #     'beta': [0.004475,0.0044765,0.004477,0.0044775,0.004525,0.00455,0.004555,0.004545],
    #     'beta2': [0.0199,0.012],
    #     'dropout' :[0.3]
    # } 
    # param_grid = {
    #     'beta': [0.06], 
    #     'dropout' :[0.4]
    # }
    param_grid = { 
        'beta': [0.01],
        'dropout' :[0.2]
    }    
    # param_grid = {
    #     'beta': [0.5,0.1,0.05,0.01],
    #     'beta2': [0.001,0.005,0.01,0.05,0.1],
    #     'dropout' :[0.3]
    # }  
    # param_grid = {
    #     'beta': [0.5,0.05,0.005],
    #     'beta2': [0.1,0.01,0.001],
    #     'dropout' :[0.3]
    # } 
    # best_score = float('-inf')
    # best_params = {'beta': None, 'beta2': None}

    for beta in param_grid['beta']:

        for dropout in param_grid['dropout']:

            # ========= Building Model =========# 
            model = trans_to_cuda(Module(hypergraphs=[HG_G, HG_L], args = opt,text_embedding=text_embedding,dropout=dropout))
            # ========= Early_stopping =========#
            save_model_path = opt.save_path + data_path+'_HGCN.pt'
            early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=save_model_path)   

            # ========= Metrics =========#
            top_K = [10, 50, 100]
            best_results = {}
            for K in top_K:
                best_results['epoch%d' % K] = [0, 0]
                best_results['metric%d' % K] = [0, 0]

            validation_history = 0.0

            for epoch in range(opt.epoch):
                total_loss, val_scores, test_scores, val_accuracy, test_accuracy = train_test(epoch, model, train_loader, val_loader, test_loader,beta)

                if validation_history <= sum(val_scores.values()):
                    validation_history = sum(val_scores.values())

                    for K in top_K:
                        test_scores['hits@' + str(K)] = test_scores['hits@' + str(K)] * 100
                        test_scores['map@' + str(K)] = test_scores['map@' + str(K)] * 100

                        best_results['metric%d' % K][0] = test_scores['hits@' + str(K)]
                        best_results['epoch%d' % K][0] = epoch
                        best_results['metric%d' % K][1] = test_scores['map@' + str(K)]
                        best_results['epoch%d' % K][1] = epoch

                    print(" -validation scores:************************************")
                    print('  - (validation) accuracy: {accu:3.3f} %'.format(accu=100 * val_accuracy))
                    # logger.info(" -validation scores:-------------------------------------\n- (validation) accuracy: {accu:3.3f} %".format(accu=100 * val_accuracy))
                    for metric in val_scores.keys():
                        print(metric + ' ' + str(val_scores[metric]* 100))
                        # logger.info(metric + ' ' + str(val_scores[metric]* 100))

                    print(" -test scores:-----------------------------------------")
                    print('  - (testing) accuracy: {accu:3.3f} %'.format(accu=100 * test_accuracy))
                    # logger.info(" -test scores:-------------------------------------\n- (testing) accuracy: {accu:3.3f} %".format(accu=100 * val_accuracy))
                    for K in top_K:
                        print('train_loss:\t%.4f\thits@%d: %.4f\tMAP@%d: %.4f\tEpoch: %d,  %d' %
                            (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                            best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
                        # logger.info('train_loss:\t%.4f\tRecall@%d: %.4f\tMAP@%d: %.4f\tEpoch: %d,  %d' %
                        #     (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                        #     best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
                print(f"\n=======================Training with beta={beta},dropout={dropout}\n")
                # logger.info(f"\n=======================Training with beta={beta}, beta2={beta2},dropout={dropout}\n")
                early_stopping(-sum(list(val_scores.values())), model)
                if early_stopping.early_stop:
                    print("\n\n\n\n\nEarly_Stoppinggggggggggg------------------------------")
                    logger.info("\n\n\n\n\nEarly_Stoppinggggggggggg------------------------------")
                    # ========= Final score =========#
                    print(" -(Finished!!) \n test scores: ")
                    print("--------------------------------------------")
                    logger.info(" -(Finished!!) \n test scores: \n--------------------------------------------")


                    for K in top_K:
                        print('hits@%d: %.4f\tMAP@%d: %.4f\tEpoch: %d, %d' %
                            (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                            best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
                        logger.info('hits@%d: %.4f\tMAP@%d: %.4f\tEpoch: %d,  %d' %
                            (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                            best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
                    print(f"=======================Training with beta={beta},dropout={dropout}\n")
                    logger.info(f"=======================Training with beta={beta},dropout={dropout}\n")
                    
                    break
if __name__ == "__main__":
    # torch.cuda.set_device(2) 
    logger,file_handler = setup_logger(opt.data_name)
    main(opt.data_name, seed=2024)

    logger.removeHandler(file_handler)
    file_handler.close()

    
