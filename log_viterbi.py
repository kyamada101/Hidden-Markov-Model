import numpy as np
#状態と観測数字からその出現確率を返す関数
def el(state,dice_num):
    if state == 'fair':
        return 1/6
    elif state == 'loaded' and int(dice_num) == 6:
        return 1/2
    else:
        return 1/10

#対数変換版Viterbi 
#観測列の長さL,隠れ状態の数K
def log_viterbi(letter):
    #viterbi変数格納用リスト
    #初期化
    V_var_list_F = [0]
    V_var_list_L = [-100000]

    #viterbi変数を格納（viterbiリスト内はすべて対数状態で保存）
    for i in range(len(letter)): #len(letter) = L
        vk_1 = V_var_list_F[-1] + np.log(0.95)
        vk_2 = V_var_list_L[-1] + np.log(0.1)
        vk_3 = V_var_list_F[-1] + np.log(0.05)
        vk_4 = V_var_list_L[-1] + np.log(0.9)

        V_var_list_F.append(np.log(el('fair',letter[i])) + max(vk_1,vk_2))
        V_var_list_L.append(np.log(el('loaded',letter[i])) + max(vk_3,vk_4))
    
    final_vk_F = V_var_list_F[-1]
    final_vk_L = V_var_list_L[-1]
    log_joint_P = max(final_vk_F, final_vk_L)

    opt_path_list = []
    if final_vk_F < final_vk_L:
        opt_path_list.append('L')
    if final_vk_L <= final_vk_F:
        opt_path_list.append('F')

    for i in range(len(letter)-1,0,-1):
        vk_1 = V_var_list_F[i] + np.log(0.95)
        vk_2 = V_var_list_L[i] + np.log(0.1)
        vk_3 = V_var_list_F[i] + np.log(0.05)
        vk_4 = V_var_list_L[i] + np.log(0.9)

        if opt_path_list[-1] == 'F' and vk_1 < vk_2:
            opt_path_list.append('L')
        elif opt_path_list[-1] == 'F' and vk_2 <= vk_1:
            opt_path_list.append('F')
        elif opt_path_list[-1] == 'L' and vk_3 < vk_4:
            opt_path_list.append('L')
        elif opt_path_list[-1] == 'L' and vk_4 <= vk_3:
            opt_path_list.append('F')
            
    opt_path = ''.join(reversed(opt_path_list))

    #返すのは対数状態の観測列・状態遷移列の同時確率、状態の最適パス
    return log_joint_P, opt_path