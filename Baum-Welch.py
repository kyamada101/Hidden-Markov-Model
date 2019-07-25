import numpy as np
#正解モデルに基づいてさいころの目を出す関数
def answer_model(prac_num):
    dice_letter = ''
    state_letter = ''
    
    state = ['fair','loaded']
    first_rate = [1/2,1/2]
    trans_fair = [0.95,0.05]
    trans_loaded = [0.1,0.9]
    
    dice = ['1','2','3','4','5','6']
    fair_weight = [1/6,1/6,1/6,1/6,1/6,1/6]
    loaded_weight = [1/10,1/10,1/10,1/10,1/10,1/2]
    
    now_state = np.random.choice(state,p = first_rate)
    for i in range(prac_num):
        if now_state == 'fair':
            dice_letter += np.random.choice(dice,p = fair_weight)
            now_state = np.random.choice(state,p = trans_fair)
            state_letter += 'F'
        elif now_state == 'loaded':
            dice_letter += np.random.choice(dice,p = loaded_weight)
            now_state = np.random.choice(state,p = trans_loaded)
            state_letter += 'L'
            
    return dice_letter, state_letter
    
#Baum-Welch用対数変換版前向きアルゴリズム
def calc_logsumexp(a,b):
    if a > b :
        return a + np.log(np.exp(b-a) + 1)
    else:
        return b + np.log(np.exp(a-b) + 1)
    
def BW_log_forward(fair_dice, loaded_dice, a_fromF, a_fromL, letter):
    #確率計算式を定義
    def el(state, dice_num):
        if state == 'fair':
            return fair_dice[int(dice_num)-1]
        if state == 'loaded':
            return loaded_dice[int(dice_num)-1]
        
    #初期化
    f_var_list_F = [0]
    f_var_list_L = [-100000]
    
    #再帰処理
    for i in range(len(letter)):
        if i == 0:
            f_var_list_F.append(np.log(el('fair',letter[i])) + calc_logsumexp((f_var_list_F[-1] + np.log(0.5)), (f_var_list_L[-1] + np.log(0.5))))
            f_var_list_L.append(np.log(el('loaded',letter[i])) + calc_logsumexp((f_var_list_F[-2] + np.log(0.5)),(f_var_list_L[-1] + np.log(0.5))))
        else:
            f_var_list_F.append(np.log(el('fair',letter[i])) + calc_logsumexp((f_var_list_F[-1] + np.log(a_fromF[0])), (f_var_list_L[-1] + np.log(a_fromL[0]))))
            f_var_list_L.append(np.log(el('loaded',letter[i])) + calc_logsumexp((f_var_list_F[-2] + np.log(a_fromF[1])),(f_var_list_L[-1] + np.log(a_fromL[1]))))


    #終了処理
    log_letter_prob = calc_logsumexp(f_var_list_F[-1], f_var_list_L[-1])

    #観測列の生起確率,前向き変数リスト（fair,loaded）（全てlog状態）
    return log_letter_prob ,f_var_list_F, f_var_list_L 

#Baum-Welch用対数変換版後ろ向きアルゴリズム
def BW_log_backward(fair_dice, loaded_dice, a_fromF, a_fromL, letter):
    #確率計算式を定義
    def el(state, dice_num):
        if state == 'fair':
            return fair_dice[int(dice_num)-1]
        if state == 'loaded':
            return loaded_dice[int(dice_num)-1]
    
    #初期化
    b_var_list_F = [0]
    b_var_list_L = [0]
    
    #再帰処理
    for i in range(len(letter)-1,0,-1):
        b_var_list_F.append(calc_logsumexp((np.log(a_fromF[0]) + np.log(el('fair',letter[i])) + b_var_list_F[-1]), ((np.log(a_fromF[1]) + np.log(el('loaded',letter[i])) + b_var_list_L[-1]))))
        b_var_list_L.append(calc_logsumexp((np.log(a_fromL[0]) + np.log(el('fair',letter[i])) + b_var_list_F[-2]), ((np.log(a_fromL[1]) + np.log(el('loaded',letter[i])) + b_var_list_L[-1]))))
    
    #最後から後ろ向き変数をリストに突っ込んでいるので前から並べ直す
    new_b_var_list_F = list(reversed(b_var_list_F))
    new_b_var_list_L = list(reversed(b_var_list_L))
    
    #終了処理
    log_letter_prob = calc_logsumexp((np.log(0.5) + np.log(el('fair',letter[0])) + new_b_var_list_F[0]), (np.log(0.5) + np.log(el('loaded',letter[0])) + new_b_var_list_L[0]))
    
    #観測列の生起確率、後ろ向き変数リスト（fair,loaded）（全てlog状態）
    return log_letter_prob, new_b_var_list_F, new_b_var_list_L

#Baum-Welchアルゴリズム
def BaumWelch(fair_dice_1st, loaded_dice_1st, a_fromF_1st, a_fromL_1st, ans_letter):
    #パラメータ初期化
    fair_dice = fair_dice_1st
    loaded_dice = loaded_dice_1st
    a_fromF = a_fromF_1st
    a_fromL = a_fromL_1st
    
    #対数尤度格納用リスト
    log_joint_P_list = []
    
    count = 0
    txt = ''
    
    while(True):
        count += 1
        #再帰
        #前向きアルゴリズム
        joint_p_A, f_var_list_F, f_var_list_L = BW_log_forward(fair_dice, loaded_dice, a_fromF, a_fromL, ans_letter)
        del f_var_list_F[0]
        del f_var_list_L[0]
        
        #後ろ向きアルゴリズム
        joint_p_B, b_var_list_F, b_var_list_L = BW_log_backward(fair_dice, loaded_dice, a_fromF, a_fromL, ans_letter)
        
        log_joint_P = joint_p_A
        
        if count == 1:
            log_joint_P_list.append(log_joint_P)

        #Ek(b)を初期化
        EF_b_1 = 0
        EL_b_1 = 0
        EF_b_2 = 0
        EL_b_2 = 0
        EF_b_3 = 0
        EL_b_3 = 0
        EF_b_4 = 0
        EL_b_4 = 0
        EF_b_5 = 0
        EL_b_5 = 0
        EF_b_6 = 0
        EL_b_6 = 0

        for i in range(len(ans_letter)):
            #Σfk(i)bk(i)を計算
            #log_forwardを使用
            log_f_multi_b_forF = f_var_list_F[i] + b_var_list_F[i] - log_joint_P
            log_f_multi_b_forL = f_var_list_L[i] + b_var_list_L[i] - log_joint_P
            
            #Ek(b)を計算（k:LF,b:1~6）
            if int(ans_letter[i]) == 1:
                EF_b_1 = calc_logsumexp(EF_b_1,log_f_multi_b_forF)
                EL_b_1 = calc_logsumexp(EL_b_1,log_f_multi_b_forL)
            if int(ans_letter[i]) == 2:
                EF_b_2 = calc_logsumexp(EF_b_2,log_f_multi_b_forF)
                EL_b_2 = calc_logsumexp(EL_b_2,log_f_multi_b_forL)
            if int(ans_letter[i]) == 3:
                EF_b_3 = calc_logsumexp(EF_b_3,log_f_multi_b_forF)
                EL_b_3 = calc_logsumexp(EL_b_3,log_f_multi_b_forL)
            if int(ans_letter[i]) == 4:
                EF_b_4 = calc_logsumexp(EF_b_4,log_f_multi_b_forF)
                EL_b_4 = calc_logsumexp(EL_b_4,log_f_multi_b_forL)
            if int(ans_letter[i]) == 5:
                EF_b_5 = calc_logsumexp(EF_b_5,log_f_multi_b_forF)
                EL_b_5 = calc_logsumexp(EL_b_5,log_f_multi_b_forL)
            if int(ans_letter[i]) == 6:
                EF_b_6 = calc_logsumexp(EF_b_6,log_f_multi_b_forF)
                EL_b_6 = calc_logsumexp(EL_b_6,log_f_multi_b_forL)
            
            if i != len(ans_letter)-1:
                #対数変換版Akl計算（k:LF,l:LF）
                A_fromF_1 = f_var_list_F[i] + np.log(a_fromF[0]) + np.log(fair_dice[int(ans_letter[i+1])-1]) + b_var_list_F[i+1] - log_joint_P
                A_fromF_2 = f_var_list_F[i] + np.log(a_fromF[1]) + np.log(loaded_dice[int(ans_letter[i+1])-1]) + b_var_list_L[i+1] - log_joint_P
                A_fromL_1 = f_var_list_L[i] + np.log(a_fromL[0]) + np.log(fair_dice[int(ans_letter[i+1])-1]) + b_var_list_F[i+1] - log_joint_P
                A_fromL_2 = f_var_list_L[i] + np.log(a_fromL[1]) + np.log(loaded_dice[int(ans_letter[i+1])-1]) + b_var_list_L[i+1] - log_joint_P
                if i == 0:
                    A_fromF0 = calc_logsumexp(0, A_fromF_1)
                    A_fromF1 = calc_logsumexp(0, A_fromF_2)
                    A_fromL0 = calc_logsumexp(0, A_fromL_1)
                    A_fromL1 = calc_logsumexp(0, A_fromL_2)
                else:
                    A_fromF0 = calc_logsumexp(A_fromF0, A_fromF_1)
                    A_fromF1 = calc_logsumexp(A_fromF1, A_fromF_2)
                    A_fromL0 = calc_logsumexp(A_fromL0, A_fromL_1)
                    A_fromL1 = calc_logsumexp(A_fromL1, A_fromL_2)
            
        #パラメータ更新
        a_fromF[0] = np.exp(A_fromF0 - calc_logsumexp(A_fromF0, A_fromF1))
        a_fromF[1] = np.exp(A_fromF1 - calc_logsumexp(A_fromF0, A_fromF1))
        a_fromL[0] = np.exp(A_fromL0 - calc_logsumexp(A_fromL0, A_fromL1))
        a_fromL[1] = np.exp(A_fromL1 - calc_logsumexp(A_fromL0, A_fromL1))
        
        EF_b_12 = calc_logsumexp(EF_b_1,EF_b_2)
        EF_b_34 = calc_logsumexp(EF_b_3,EF_b_4)
        EF_b_56 = calc_logsumexp(EF_b_5,EF_b_6)
        EF_b_1234 = calc_logsumexp(EF_b_12,EF_b_34)
        EF_b_123456 = calc_logsumexp(EF_b_1234,EF_b_56)
        
        EL_b_12 = calc_logsumexp(EL_b_1,EL_b_2)
        EL_b_34 = calc_logsumexp(EL_b_3,EL_b_4)
        EL_b_56 = calc_logsumexp(EL_b_5,EL_b_6)
        EL_b_1234 = calc_logsumexp(EL_b_12,EL_b_34)
        EL_b_123456 = calc_logsumexp(EL_b_1234,EL_b_56)
        
        
        fair_dice[0] = np.exp(EF_b_1 - EF_b_123456)
        loaded_dice[0] = np.exp(EL_b_1 - EL_b_123456)
        fair_dice[1] = np.exp(EF_b_2 - EF_b_123456)
        loaded_dice[1] = np.exp(EL_b_2 - EL_b_123456)
        fair_dice[2] = np.exp(EF_b_3 - EF_b_123456)
        loaded_dice[2] = np.exp(EL_b_3 - EL_b_123456)
        fair_dice[3] = np.exp(EF_b_4 - EF_b_123456)
        loaded_dice[3] = np.exp(EL_b_4 - EL_b_123456)
        fair_dice[4] = np.exp(EF_b_5 - EF_b_123456)
        loaded_dice[4] = np.exp(EL_b_5 - EL_b_123456)
        fair_dice[5] = np.exp(EF_b_6 - EF_b_123456)
        loaded_dice[5] = np.exp(EL_b_6 - EL_b_123456)
            
        #対数尤度計算
        if count > 1 and np.abs(log_joint_P - log_joint_P_list[-1]) < 0.00001:
            break
        if count != 1:
            log_joint_P_list.append(log_joint_P)
        #最大1000回で打ち切り
        if count == 1000:
            print('this model do not end for only {} times...'.format(count))
            break
        #10回ごとにパラメータの推移を表示
        if count % 10 == 0:
            print('================================================================')
            print('EMアルゴリズム {} 回目の試行'.format(count))
            print('fair dice　の確率は \n {}'.format(fair_dice))
            print('loaded dice　の確率は \n {}'.format(loaded_dice))
            print('fair diceからの遷移確率は \n {}'.format(a_fromF))
            print('loaded diceからの遷移確率は \n {}'.format(a_fromL))
            
            #後でtxtファイルに書き込むときに作る文章
            txt += '================================================================\n'
            txt += 'EMアルゴリズム {} 回目の試行\n'.format(count)
            txt += 'fair dice　の確率は \n'
            txt += '1:{0} 2:{1} 3:{2} 4:{3} 5:{4} 6:{5}\n'.format(fair_dice[0],fair_dice[1],fair_dice[2],fair_dice[3],fair_dice[4],fair_dice[5])
            txt += 'loaded dice　の確率は \n'
            txt += '1:{0} 2:{1} 3:{2} 4:{3} 5:{4} 6:{5}\n'.format(loaded_dice[0],loaded_dice[1],loaded_dice[2],loaded_dice[3],loaded_dice[4],loaded_dice[5])
            txt += 'fair diceからの遷移確率は \n'
            txt += 'fair⇒fair:{0} fair⇒Loaded:{1}\n'.format(a_fromF[0],a_fromF[1])
            txt += 'loaded diceからの遷移確率は \n'
            txt += 'loaded⇒fair:{0} loaded⇒loaded:{1}\n'.format(a_fromL[0],a_fromL[1])
        
    #さいころの確率（公正・不正）、状態の遷移確率（公正、不正）、同時確率リスト、EMアルゴリズムの手続きを踏んだ回数、書き込み用テキスト
    return fair_dice, loaded_dice, a_fromF, a_fromL, log_joint_P_list, count, txt

#対数変換版Viterbiアルゴリズム
#観測列の長さL,隠れ状態の数K
def log_viterbi(letter,fair_dice,loaded_dice,p_fair,p_loaded):
    #確率計算式を定義
    def el(state, dice_num):
        if state == 'fair':
            return fair_dice[int(dice_num)-1]
        if state == 'loaded':
            return loaded_dice[int(dice_num)-1]

    #viterbi変数格納用リスト
    #初期化
    V_var_list_F = [0]
    V_var_list_L = [-100000]

    #再帰
    for i in range(len(letter)): #len(letter) = L
        vk_1 = V_var_list_F[-1] + np.log(p_fair[0])
        vk_2 = V_var_list_L[-1] + np.log(p_loaded[0])
        vk_3 = V_var_list_F[-1] + np.log(p_fair[1])
        vk_4 = V_var_list_L[-1] + np.log(p_loaded[1])

        V_var_list_F.append(np.log(el('fair',letter[i])) + max(vk_1,vk_2))
        V_var_list_L.append(np.log(el('loaded',letter[i])) + max(vk_3,vk_4))
    
    final_vk_F = V_var_list_F[-1]
    final_vk_L = V_var_list_L[-1]
    joint_P = max(final_vk_F, final_vk_L)

    opt_path_list = []
    if final_vk_F < final_vk_L:
        opt_path_list.append('L')
    if final_vk_L <= final_vk_F:
        opt_path_list.append('F')

    for i in range(len(letter)-1,0,-1):
        vk_1 = V_var_list_F[i] + np.log(p_fair[0])
        vk_2 = V_var_list_L[i] + np.log(p_loaded[0])
        vk_3 = V_var_list_F[i] + np.log(p_fair[1])
        vk_4 = V_var_list_L[i] + np.log(p_loaded[1])

        if opt_path_list[-1] == 'F' and vk_1 < vk_2:
            opt_path_list.append('L')
        elif opt_path_list[-1] == 'F' and vk_2 <= vk_1:
            opt_path_list.append('F')
        elif opt_path_list[-1] == 'L' and vk_3 < vk_4:
            opt_path_list.append('L')
        elif opt_path_list[-1] == 'L' and vk_4 <= vk_3:
            opt_path_list.append('F')

    opt_path = ''.join(reversed(opt_path_list))

    return joint_P, opt_path

#Viterbiによる最適パスと実際のパスの比較
def compare(state,path):
    if len(state) != len(path):
        print('length do not match...')
        return 0
    count = 0
    for i in range(len(state)):
        if state[i] == path[i]:
            count += 1
    acc = count / len(state)
    return acc

#上記を使ってBaum-Welchを実行、結果を書き込んだテキストを作成
with open('./Bioinfo-result.txt','w')as f:
    i = 300 #生成する観測列の長さ
    dice, state = answer_model(i)
    fair_dice = np.random.rand(6)
    fair_dice /= sum(fair_dice)
    loaded_dice = np.random.rand(6)
    loaded_dice /= sum(loaded_dice)
    a_fromF = np.random.rand(2)
    a_fromF /= sum(a_fromF)
    a_fromL = np.random.rand(2)
    a_fromL /= sum(a_fromL)
    f.write('観測列の長さ{}で実行\n'.format(i))
    f.write('正解モデルによる観測列は\n{}\n'.format(dice))
    f.write('正解モデルによる状態遷移列は\n{}\n'.format(state))
    f.write('======================================================================\n')
    f.write('fair diceの確率の初期値は\n')
    f.write('1:{0} 2:{1} 3:{2} 4:{3} 5:{4} 6:{5}\n'.format(fair_dice[0],fair_dice[1],fair_dice[2],fair_dice[3],fair_dice[4],fair_dice[5]))
    f.write('loaded diceの確率の初期値は\n')
    f.write('1:{0} 2:{1} 3:{2} 4:{3} 5:{4} 6:{5}\n'.format(loaded_dice[0],loaded_dice[1],loaded_dice[2],loaded_dice[3],loaded_dice[4],loaded_dice[5]))
    f.write('fair dice からの遷移確率の初期値は\n')
    f.write('fair⇒fair:{0} fair⇒Loaded:{1}\n'.format(a_fromF[0],a_fromF[1]))
    f.write('loaded dice からの遷移確率の初期値は\n')
    f.write('loaded⇒fair:{0} loaded⇒loaded:{1}\n'.format(a_fromL[0],a_fromL[1]))
    fair,loaded,p_fair,p_loaded,P_list,count,txt = BaumWelch(fair_dice, loaded_dice, a_fromF, a_fromL, dice)
    f.write(txt)
    f.write('======================================================================\n')
    f.write('最終的な収束値は以下\n')
    f.write('EMアルゴリズム {} 回で収束\n'.format(count))
    f.write('fair dice　の確率は \n')
    f.write('1:{0} 2:{1} 3:{2} 4:{3} 5:{4} 6:{5}\n'.format(fair[0],fair[1],fair[2],fair[3],fair[4],fair[5]))
    f.write('loaded dice　の確率は \n')
    f.write('1:{0} 2:{1} 3:{2} 4:{3} 5:{4} 6:{5}\n'.format(loaded[0],loaded[1],loaded[2],loaded[3],loaded[4],loaded[5]))
    f.write('fair diceからの遷移確率は \n')
    f.write('fair⇒fair:{0} fair⇒Loaded:{1}\n'.format(p_fair[0],p_fair[1]))
    f.write('loaded diceからの遷移確率は \n')
    f.write('loaded⇒fair:{0} loaded⇒loaded:{1}\n'.format(p_loaded[0],p_loaded[1]))
    
    print('===============================================================')
    print('following is final result')
    print('fair dice probability is following \n {}'.format(fair))
    print('loaded dice probability is following \n {}'.format(loaded))
    print('transiton probability from fair is follwoing \n {}'.format(p_fair))
    print('transiton probability from loaded is follwoing \n {}'.format(p_loaded))
    print('EM phase is {} times passed'.format(count))
    
    joint_p, opt_path = log_viterbi(dice,fair,loaded,p_fair,p_loaded)
    acc = compare(state,opt_path)
    print('最適パスの正解率は{}'.format(acc))
    f.write('======================================================================\n')
    f.write('推定パラメータをもとにしたviterbiアルゴリズムによる最適パスの正解率は{}\n'.format(acc))
    f.write('Viterbiアルゴリズムによる最適パスは\n{}'.format(opt_path))