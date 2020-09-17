'''
Created on Nov 14, 2019

@author: Sascha Hornauer
'''
from MDistance import Distance_Eval
from PIL import Image
import os
import pandas as pd
import numpy as np
import cv2

def calc_to_csv(root_path,csv_filename,evaluator):
    
    distances = []
    
    for root, dirs, files in os.walk(root_path,followlinks=True):
        print ("Traversing {}".format(root))
        for file_ in files:
            filename = os.path.join(root, file_) 
            print (file_)
            if filename.endswith('jpg'):
                with open(filename, 'rb') as f:
                    img = Image.open(f)                    
                    distance, features = evaluator.get_distance_and_features(img)
                    distances.append({'filename':filename, 'distance':distance, 'features':features.squeeze()})            

    distances_pd = pd.DataFrame(distances)
    distances_pd.to_csv(csv_filename, sep=',',index=False)
    
if __name__ == '__main__':

    # Checkpoint
    evaluator = Distance_Eval('/home/saschaho/Simcenter/lemniscate.pytorch/checkpoint.pth.tar')
    #evaluator = Distance_Eval('/home/saschaho/Simcenter/lemniscate.pytorch/lemniscate_resnet50.pth.tar')

    csv_filename = '/home/saschaho/Simcenter/lemniscate.pytorch/distances.csv'
    
    root_path = '/home/saschaho/Simcenter/copy_for_zhirong/Florida/'
    recalculate_distances = True
    if recalculate_distances: calc_to_csv(root_path,csv_filename,evaluator)
    
    pkl_save_path = '/home/saschaho/Simcenter/copy_for_zhirong/pkl_stat_data'
    pkl_stats_file = '/home/saschaho/Simcenter/copy_for_zhirong/stats.pkl'
    
    ### Calculate and show distance images
    distances = pd.read_csv(csv_filename)
     
    closest_ten = distances.sort_values(by='distance').iloc[0:50]
    furthest_ten = distances.sort_values(by='distance').iloc[-70:]
     
    for i, image in enumerate(furthest_ten.iterrows()):
        image = image[1] # Some index number is at position 0
        img_path = image['filename']
        img_name = os.path.basename(img_path).split('.')[0]
         
        print ("Processing {}".format(img_path))
        img_default_dist = image['distance']
                          
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img_array = np.array(img)

        final_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        #dist_at_pos_pd = evaluator.image_check(img_array, mode='peephole')
         
        #final_img = annotate_img(img_array=img_array, dist_at_pos_pd=dist_at_pos_pd, default_dist=img_default_dist)
        #cv2.imwrite('furthest_{}.png'.format(i+50),final_img)
        cv2.imshow('test',final_img)
        cv2.waitKey(1000)
        print (i)
          
        
