import pandas as pd
import os
def generate(root_dir, is_train=True):
    df = pd.DataFrame(columns=['dir', 'class'])
    if(is_train):
        for d in range(1,7):
            cur_cat = str(d+5000)
            cur_dir = os.path.join(root_dir, cur_cat)
            for l in os.listdir(cur_dir):
                df.loc[len(df)] = [os.path.join(cur_dir, l), cur_cat]
        df.to_csv(os.path.join(root_dir, 'description_train.csv'))
    else:
        for d in range(1,7):
            cur_cat = str(d+5000)
            cur_dir = os.path.join(root_dir, cur_cat)
            for l in os.listdir(cur_dir):
                df.loc[len(df)] = [os.path.join(cur_dir, l), cur_cat]
        df.to_csv(os.path.join(root_dir, 'description_test.csv'))


generate('/data/sascha/Simcenter/cleaned_images/images/train')
generate('/data/sascha/Simcenter/cleaned_images/images/validate', is_train=False)
generate('/data/sascha/Simcenter/cleaned_images/test', is_train=False)