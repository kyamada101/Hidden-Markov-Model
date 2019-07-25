#正解モデルに基づいてさいころの目を出す関数
import numpy as np

def answer_model(prac_num):
    dice_letter = ''
    state_letter = ''
    
    #モデルの状態・遷移確率を指定
    state = ['fair','loaded']
    first_rate = [1/2,1/2]
    trans_fair = [0.95,0.05]
    trans_loaded = [0.1,0.9]
    
    #状態ごとの事象確率を指定（今回はさいころの出る目の確率）
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

    #返すのはprac_num個文の観測列、状態遷移列      
    return dice_letter, state_letter