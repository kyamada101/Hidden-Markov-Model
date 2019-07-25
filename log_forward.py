import numpy as np
#状態と観測数字からその出現確率を返す関数
def el(state,dice_num):
    if state == 'fair':
        return 1/6
    elif state == 'loaded' and int(dice_num) == 6:
        return 1/2
    else:
        return 1/10

#対数変換版前向きアルゴリズム
def calc_logsumexp(a,b):
    if a > b :
        return a + np.log(np.exp(b-a) + 1)
    else:
        return b + np.log(np.exp(a-b) + 1)

def log_forward(letter):
    #初期化
    f_var_list_F = [0]
    f_var_list_L = [-100000]
    
    #再帰処理
    for i in range(len(letter)):
        if i == 0:
            f_var_list_F.append(np.log(el('fair',letter[i])) + calc_logsumexp((f_var_list_F[-1] + np.log(0.5)), (f_var_list_L[-1] + np.log(0.5))))
            f_var_list_L.append(np.log(el('loaded',letter[i])) + calc_logsumexp((f_var_list_F[-2] + np.log(0.5)),(f_var_list_L[-1] + np.log(0.5))))
        else:
            f_var_list_F.append(np.log(el('fair',letter[i])) + calc_logsumexp((f_var_list_F[-1] + np.log(0.95)), (f_var_list_L[-1] + np.log(0.1))))
            f_var_list_L.append(np.log(el('loaded',letter[i])) + calc_logsumexp((f_var_list_F[-2] + np.log(0.05)),(f_var_list_L[-1] + np.log(0.9))))
                  
        
    #終了処理
    log_letter_prob = calc_logsumexp(f_var_list_F[-1], f_var_list_L[-1])
    
    #返すのは観測列の生起確率、前向き変数リスト
    return np.exp(log_letter_prob) ,f_var_list_F,f_var_list_L #観測列の生起確率,前向き変数リスト（fair,loaded）