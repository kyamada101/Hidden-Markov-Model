import numpy as np

#状態と観測数字からその出現確率を返す関数
def el(state,dice_num):
    if state == 'fair':
        return 1/6
    elif state == 'loaded' and int(dice_num) == 6:
        return 1/2
    else:
        return 1/10

#前向きアルゴリズム
#観測列が引数
def forward(letter):
    #初期化
    f_var_list_F = [1]
    f_var_list_L = [0]
    
    #再帰処理
    for i in range(len(letter)):
        if i == 0:
            f_var_list_F.append(el('fair',letter[i]) * (f_var_list_F[-1] * 0.5 + f_var_list_L[-1] * 0.5))
            f_var_list_L.append(el('loaded',letter[i]) * (f_var_list_L[-1] * 0.5 + f_var_list_F[-2] * 0.5))
        else:
            f_var_list_F.append(el('fair',letter[i]) * (f_var_list_F[-1] * 0.95 + f_var_list_L[-1] * 0.1))
            f_var_list_L.append(el('loaded',letter[i]) * (f_var_list_L[-1] * 0.9 + f_var_list_F[-2] * 0.05))
        
    #終了処理
    letter_prob = f_var_list_F[-1] + f_var_list_L[-1]
    
    #返すのは観測列の生起確率、前向き変数リスト
    return letter_prob, f_var_list_F, f_var_list_L