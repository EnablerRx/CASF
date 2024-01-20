import pickle
import random
import numpy as np
import pandas as pd
from scipy import stats 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler



def ActiveSampling(metric_num, scores, hmetric_key, pre_sample_rate, batch_sample_rate, batch_inc_num):
    '''
        Input：
            metric_num：pre-ranked metric
            scores：human evaluation scores
            hmetric_key：human evaluation aspect
            pre_sample_rate：pre-sampling rate eg. input 0.5 for 50%
            batch_sample_rate：sampling rate for each batch
            batch_inc_num：increasing batch number
            control_start_center：1 center，0 uncenter
    '''
    
    def ngramSimilarity(sent1, sent2):
        '''
            Function: 
                Calculate the ngram similarity, the larger the less similar, the smaller the more similar
        '''
        def ngrams(sentence, n):
            return list(zip(*[sentence.split()[i:] for i in range(n)]))

        n = 2
        ng1 = ngrams(sent1, n)
        ng2 = ngrams(sent2, n)
        ng_union = list(set(ng1) & set(ng2)) 
        ng_simi = len(ng1)+len(ng2)-2*(len(ng_union))
        return ng_simi
    
    def NGramContrainedSampling(dataset_op, final_sample_op, start, end, batch_sample_list):
        ''' 
            Function: 
                Determine the coincidence degree between the current sample and the previous samples, take a sample from a given range of samples, and return the sample subscript
            Process：
                Calculate samples that meet the required similarity from left to right. If found, loop break and return; if not, increase the repeatability requirement and continue looking.
                There are two steps to determine the repeatability: the first step is what has been picked in the previous round, and the second round is what has been picked in the current round.
            Input：
                dataset_op: All the samples already taken
                            Dimensions: Number of samples x number of systems x 1
                final_sample_op: The remaining samplable copies after sorting (including those collected in the current round). 
                            Dimensions: number of samples x number of systems
                start: Left interval
                end: Right interval (not optional)
                batch_sample_list: Samples that has been taken this round
            Output：
                idx: Returns the selected sample subscript
        '''
        largestNGramControl = 10000000 # The upper limit of ngram coincidence degree
        # Calculate ngram and save the results
        ngram_final_sample_op = [[0 for _ in range(len(final_sample_op[0]))] for _ in range(len(final_sample_op))]
        for idx in range(start,end):                                                      
            for s in range(len(final_sample_op[0])):                                      
                for sample in range(len(dataset_op)):                                     
                    ngram_temp = ngramSimilarity(dataset_op[sample], final_sample_op[idx][s])
                    if ngram_temp > ngram_final_sample_op[idx][s]:
                        ngram_final_sample_op[idx][s] = ngram_temp

        for i in range(largestNGramControl):                                              
            for idx in range(start,end):                                                   

                # First, determine the degree of duplication in the samples already taken
                num_samplepass = 0                                                         
                for sample in range(len(dataset_op)):                                       
                    num_syspass = 0                                                         
                    for s in range(len(final_sample_op[0])):                                
                        if ngram_final_sample_op[idx][s] <= i:                              
                            num_syspass += 1
                        else:                                                               
                            break
                    if num_syspass == len(final_sample_op[0]): # All samples of the system passed the dataset pool repeatability test
                        num_samplepass += 1
                # Secondly, we judge the degree of repetition in the current round of samples
                if num_samplepass == len(dataset_op): # all samples pass the first checking, go on to the second phase checking
                    num_own_sys_num = 0
                    for s in range(len(final_sample_op[0])):
                        num_syspass = 0
                        for s_last in range(len(final_sample_op[0])):                       
                            num_samplepass = 0
                            for sample_last in batch_sample_list:
                                if ngramSimilarity(final_sample_op[idx][s],final_sample_op[sample_last][s_last]) <= i:
                                    num_samplepass += 1
                                else:
                                    break
                            if num_samplepass == len(batch_sample_list):
                                num_syspass += 1
                        if num_syspass == len(final_sample_op[0]):
                            num_own_sys_num += 1
                    if num_own_sys_num == len(final_sample_op[0]):  # all systems pass
                        return [idx]



    '''
        Data Preprocessing
    ''' 
    scores_key = scores.keys()
    sys_key = scores[list(scores_key)[0]]['sys_summs'].keys()  # name of NLG systems
    sys_num = len(sys_key)  # system numbers
    ametric_key = ['bert_score_f', 'mover_score', 'rouge1_f', 'rouge2_f', 'rougel_f', 
                'bart_score_avg_f', 'BLEU_4', 'METEOR'] # automatic metrics pool
    metric_key = ametric_key + hmetric_key
    score_all = []                                                                     
    sysop_all = []                                                                     
    for i in range(len(scores)):                                                         
        temp_sys = []                                                
        temp_op = []                               
        for j in range(len(sys_key)):         
            temp = []                 
            for k in range(len(metric_key)): 
                if k == 0:
                    temp.append(float(scores[list(scores_key)[i]]['sys_summs'][list(sys_key)[j]]['scores'][metric_key[k]]))
                else:
                    temp.append(scores[list(scores_key)[i]]['sys_summs'][list(sys_key)[j]]['scores'][metric_key[k]])
            temp_sys.append(temp)
            temp_op.append(scores[list(scores_key)[i]]['sys_summs'][list(sys_key)[j]]['sys_summ'])
        score_all.append(temp_sys)                                                     
        sysop_all.append(temp_op)

    '''
        Preliminary Sampling:
            Function:
                Pre-sort the data and sort it by reference to the automatic metric
                Sampling some samples in advance
            Output:
                pre_sample_list: Presampling sample subscript
                dataset_dele: The rest dataset 
    '''
    # Pre-sort the data and sort it by reference to the automatic metric
    for i in range(len(score_all)-1):         
        for j in range(i+1, len(score_all)):         
            sum1 = 0
            sum2 = 0
            for k in range(len(score_all[0])):
                sum1 += score_all[i][k][metric_num]
                sum2 += score_all[j][k][metric_num]
                if sum1 > sum2:                              
                    temp = np.array(score_all[i])             
                    score_all[i] = score_all[j]
                    score_all[j] = temp
                    sysop_all[i], sysop_all[j] = sysop_all[j], sysop_all[i] 

    # Sampling some samples in advance
    population = len(score_all) # sample numbers
    insta_part = pre_sample_rate  # preliminary sampling rate
    sys_sample_size = int(population*insta_part)  # sampling size
    step = int(population/sys_sample_size)                                                 
    start_centered = 0 
    pre_sample_list = np.linspace(start_centered,population,num=sys_sample_size,endpoint=False,dtype=int) 

    dataset_pre = np.array(score_all).transpose((1, 0, 2))          
    dataset = []                                    
    dataset_dele = []                               
    dataset_pre_op = list(map(list, zip(*sysop_all))) 

    dataset_op = []
    dataset_dele_op = []
    for i in range(len(dataset_pre)):                   
        for j in pre_sample_list:
            dataset.append(dataset_pre[i][j])
            dataset_op.append(dataset_pre_op[i][j])
        temp_dataset = np.delete(dataset_pre[i], pre_sample_list, 0)    
        temp_dataset_op = np.delete(dataset_pre_op[i], pre_sample_list, 0)
        dataset_dele.append(temp_dataset)
        dataset_dele_op.append(temp_dataset_op)
    dataset = np.array(dataset)

    '''
        Batch Sampling
            Function：Batch sampling 
            Variable：
                dataset: the Selected samples
                dataset_dele: The rest samples
                X: Automatic scores
                Y: Human score
    '''
    for i in range(batch_inc_num): 
        X = []
        X = dataset[:,0:8]
        transfer = StandardScaler()
        dataset_hscore = []
        for hm in range(len(hmetric_key)):
            dataset_hscore.append(transfer.fit_transform(dataset[:,8+hm].reshape(-1,1)).ravel())
        Y = []
        for i in range(len(dataset)):
            sum_temp = 0
            for j in range(len(hmetric_key)):
                sum_temp += dataset_hscore[j][i]
            Y.append(sum_temp)

        '''
            Learner Training:
                Function:
                    Train the learner to predict human score
                Input:
                    X: Automatic scores  
                    Y: Human score 
        '''
        regressor = GradientBoostingRegressor(max_depth=4, 
                    n_estimators=200,
                    random_state=2)
        regressor.fit(X, Y)

        '''
            Learner Testing:
                Function:
                    Calculate learner prediction scores for all samples
                Output:
                    sample_list: Sampling list
        '''
        # Calculate learner prediction scores for all samples
        test_dataset = np.array(dataset_dele)
        sample_res_test = []
        for i in range(len(test_dataset)):                       
            X_test = test_dataset[i]                     
            X_test = X_test[:,0:8]
            y_pred_pers = regressor.predict(X_test)
            sample_res_test.append(np.column_stack((test_dataset[i], y_pred_pers)))  

        # Sort samples by learner prediction scores
        final_sample = np.array(sample_res_test).transpose((1, 0, 2))      
        final_sample_op = np.array(dataset_dele_op).transpose((1, 0))
        metric_num = len(metric_key) 
        for i in range(len(final_sample)-1):  
            for j in range(i+1, len(final_sample)):    
                sum1 = 0
                sum2 = 0
                for k in range(len(final_sample[0])):
                    sum1 += final_sample[i][k][metric_num]
                    sum2 += final_sample[j][k][metric_num]
                if sum1 > sum2:
                    temp = np.array(final_sample[i])
                    final_sample[i] = final_sample[j]
                    final_sample[j] = temp
                    final_sample_op[i], final_sample_op[j] = final_sample_op[j], final_sample_op[i] 

        # Sampling
        batch_sample_list = [] # selected samples
        population_sub = len(final_sample) # total number of samples for batch sampling
        insta_part = batch_sample_rate # sampling rate of each batch
        sys_sample_size = int(population*insta_part) # sampling size

        start_centered = 0 
        start = 0
        end = 1
        ini_sample_list = np.linspace(start_centered,population_sub,num=sys_sample_size,endpoint=False,dtype=int) 
        ini_sample_list = np.append(ini_sample_list, population_sub) 
        while len(batch_sample_list) < sys_sample_size:
            batch_sample_list += NGramContrainedSampling(dataset_op, final_sample_op, ini_sample_list[start], ini_sample_list[end], batch_sample_list) 
            start = start + 1
            end = end + 1
        # print("sample list:")
        # print(batch_sample_list)


        dataset = list(dataset)
        dataset_op = list(dataset_op)
        dataset_dele_batch = [] # rest samples fo each batch
        dataset_dele_batch_op = []
        dataset_pre_batch = np.array(final_sample).transpose((1, 0, 2))
        dataset_pre_batch_op = np.array(final_sample_op).transpose((1, 0))
        for i in range(len(dataset_pre_batch)):              
            for j in batch_sample_list:
                dataset.append(dataset_pre_batch[i][j][0:len(metric_key)])
                dataset_op.append(dataset_pre_batch_op[i][j])
            temp_dataset = np.delete(dataset_pre_batch[i], batch_sample_list, 0)  
            temp_dataset_op = np.delete(dataset_pre_batch_op[i], batch_sample_list, 0)
            dataset_dele_batch.append(temp_dataset)
            dataset_dele_batch_op.append(temp_dataset_op)
        dataset_dele = np.array(dataset_dele_batch)           
        dataset_dele_op = np.array(dataset_dele_batch_op)
        dataset = np.array(dataset)        
        dataset_op = np.array(dataset_op)


    # Organize the dataset by system:
    # dataset dimension：(System dimension * Number of batches * Number of samples per batch)*1
    # dataset_res dimensions: Number of systems x number of samples x number of metric
    dataset_res = [[] for _ in range(sys_num)]
    index = 0
    for i in range(1 + batch_inc_num):
        for s in range(sys_num):
            for j in range(sys_sample_size):
                dataset_res[s].append(dataset[index])
                index += 1
    dataset_sys_res = np.mean(dataset_res,1)


    # Each human metric score is calculated according to the system. Dimension: Number of systems x number of human metrics
    total_res = [] # Human evaluation results for each system. Dimension: Number of systems x number of human metrics
    for i in range(sys_num):
        temp_sys = []
        for k in range(len(scores)):
            temp = []
            for j in range(len(hmetric_key)):
                temp.append(scores[list(scores_key)[k]]['sys_summs'][list(sys_key)[i]]['scores'][hmetric_key[j]])
            temp_sys.append(temp)
        total_res.append(temp_sys)
    total_res_mean = np.mean(total_res,1)

    tau = []
    for m in range(len(hmetric_key)):
        sum_sys = []
        for i in range(len(dataset_sys_res)):
            sum_sys.append(dataset_sys_res[i][8+m]) 
        sys_order = np.argsort(sum_sys)
        sum_total = []
        for i in range(len(total_res_mean)):
            sum_total.append(total_res_mean[i][m])
        total_order = np.argsort(sum_total)
        tau_value, p_value = stats.kendalltau(total_order, sys_order)
        tau.append(tau_value)
    print(tau)
    return tau


## dataset
dir_Newsroom = './DatasetPrePro/SUM/Newsroom/scores.pkl'
dir_REALSumm = './DatasetPrePro/SUM/REALSumm/scores.pkl'
dir_SummEval = './DatasetPrePro/SUM/SummEval/scores_bleu.pkl'
dir_OpenAI_tldraxis1 = './DatasetPrePro/SUM/OpenAI/scores_OpenAI_tldraxis1.pkl'
dir_OpenAI_tldraxis2 = './DatasetPrePro/SUM/OpenAI/scores_OpenAI_tldraxis2.pkl'
dir_OpenAI_cnndm1 = './DatasetPrePro/SUM/OpenAI/scores_OpenAI_cnndm1.pkl'
dir_OpenAI_cnndm3 = './DatasetPrePro/SUM/OpenAI/scores_OpenAI_cnndm3.pkl'
dir_zhen2020 = './DatasetPrePro/MT/newstest2020/scores_zhen2020.pkl'
dir_DialSummEval = './DatasetPrePro/SUM/DialSummEval/scores_DialSummEval.pkl'
dir_ende2020 = './DatasetPrePro/MT/newstest2020/scores_ende2020.pkl'
dir_zhen2021 = './DatasetPrePro/MT/newstest2021/scores_zhen2021.pkl'
dir_PersonaChat = './DatasetPrePro/DG/scores_PersonaChat.pkl'
dir_SG_mansRoc = './DatasetPrePro/StoryGen/scores_mans_roc.pkl'
dir_SG_mansWP = './DatasetPrePro/StoryGen/scores_mans_wp.pkl'
dir_MMGen_THUMB_CNNDM = './DatasetPrePro/MultiModalGen/scores_THUMB_cnndm.pkl'
dir_MMGen_VATEX = './DatasetPrePro/MultiModalGen/scores_VATEX.pkl'
dir_scores = [dir_SummEval, dir_REALSumm, dir_Newsroom, dir_DialSummEval,
                dir_OpenAI_tldraxis1, dir_OpenAI_tldraxis2, dir_OpenAI_cnndm1, dir_OpenAI_cnndm3,
                dir_ende2020, dir_zhen2020, dir_zhen2021, 
                dir_PersonaChat,
                dir_SG_mansRoc, dir_SG_mansWP,
                dir_MMGen_THUMB_CNNDM, dir_MMGen_VATEX]

## Human evaluation aspects
hmetric_key_Newsroom = ['coherence', 'fluency', 'informativeness', 'relevance']   
hmetric_key_DialSummEval = ['consistency','relevance','fluency','coherence'] 
hmetric_key_SummEval = ['coherence', 'consistency', 'fluency', 'relevance'] 
hmetric_key_REALSumm = ['litepyramid_recall']  
hmetric_key_OpenAI = ['accuracy','coherence','coverage','overall'] 
hmetric_key_zhen2021 = ['mqm']  
hmetric_key_ende2020 = ['mqm','psqm']  
hmetric_key_zhen2020 = ['mqm','psqm']
hmetric_key_PersonaChat = ['Understandable','Natural','Maintains Context', 'Engaging', 'Uses Knowledge', 'Overall'] 
hmetric_key_SG_mansRoc = ['overall']
hmetric_key_SG_mansWP = ['overall']
hmetric_key_MMGen_THUMB_MSCOCO = ["human_score"]
hmetric_key_MMGen_VATEX = ["human_metric1", "human_metric2", "human_metric3"]
hmetric_keys = [hmetric_key_SummEval, hmetric_key_REALSumm, hmetric_key_Newsroom, hmetric_key_DialSummEval,
                hmetric_key_OpenAI, hmetric_key_OpenAI, hmetric_key_OpenAI, hmetric_key_OpenAI, hmetric_key_OpenAI,
                hmetric_key_ende2020, hmetric_key_zhen2020, hmetric_key_zhen2021,
                hmetric_key_PersonaChat,
                hmetric_key_SG_mansRoc, hmetric_key_SG_mansWP,
                hmetric_key_MMGen_THUMB_MSCOCO,hmetric_key_MMGen_VATEX]



sample_rate = 0.5 # total sampling rate
res = []
AS_pre_sample_rate = 0.1 # preliminary sampling rate
AS_batch_sample_rate = 0.1 # sampling rate of each batch
AS_batch_inc_num = 4 # batch number
for dir_score,hmetric_key in zip(dir_scores,hmetric_keys): 
    scores = pickle.load(open(dir_score, 'rb'), encoding='utf-8')
    AS_temp_res_perit = [] 
    for metric_num in range(0,8):   
        AS_temp_res_perit.append(ActiveSampling(metric_num, scores, hmetric_key, AS_pre_sample_rate, AS_batch_sample_rate, AS_batch_inc_num))
    AS_temp_res_perit = list(map(list, zip(*AS_temp_res_perit))) 
    for i in range(len(AS_temp_res_perit)): 
        res.append(AS_temp_res_perit[i])
    df = pd.DataFrame(res)
    df.to_excel("./result/res.xlsx", index=False)
df = pd.DataFrame(res)
df.to_excel("./result/res.xlsx", index=False)
