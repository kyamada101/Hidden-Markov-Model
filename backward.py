import numpy as np
#状態と観測数字からその出現確率を返す関数
def el(state,dice_num):
    if state == 'fair':
        return 1/6
    elif state == 'loaded' and int(dice_num) == 6:
        return 1/2
    else:
        return 1/10

#後ろ向きアルゴリズム
def backward(letter):
    #初期化
    b_var_list_F = [1]
    b_var_list_L = [1]
    
    #再帰処理
    for i in range(len(letter)-1,0,-1):
        b_var_list_F.append((0.95 * el('fair',letter[i]) * b_var_list_F[-1]) + (0.05 * el('loaded',letter[i]) * b_var_list_L[-1]))
        b_var_list_L.append((0.1 * el('fair',letter[i]) * b_var_list_F[-2]) + (0.9 * el('loaded',letter[i]) * b_var_list_L[-1]))

    #最後から後ろ向き変数をリストに突っ込んでいるので前から並べ直す
    new_b_var_list_F = list(reversed(b_var_list_F))
    new_b_var_list_L = list(reversed(b_var_list_L))
    
    #終了処理
    letter_prob = (0.5 * el('fair',letter[0]) * new_b_var_list_F[0]) + (0.5 * el('loaded',letter[0]) * new_b_var_list_L[0])
    
    #返すのは観測列の生起確率、後ろ向き変数リスト（fair,loaded）
    return letter_prob, new_b_var_list_F, new_b_var_list_L 