# -*- coding: utf-8 -*-
"""
@author: Nathan Cai
"""
from deepface import DeepFace
import pandas as pd
import numpy as np
import matplotlib as plt
import util

def main():
    n = 1000
    root = './data/'
    
    val =  pd.read_csv('./data/fairface_label_val.csv')
    test =  pd.read_csv('./data/fairface_label_train.csv')
    
    data = util.adjust_csv(val)

    hld_gend = []
    hld_race = []
    hld_age = []
    ff_gend = np.array(data.gender[:n])
    ff_race = np.array(data.race[:n])
    ff_age = np.array(data.age[:n])
    retu_gend = []
    retu_race = []
    
    genders = list(set(ff_gend))
    races = list(set(ff_race))
    ages_ = list(set(ff_age))
    
    df = data[:n]
    df = df.reset_index()
    #print(df)
    
    for index, row in df.iterrows():
        try:
            tmp = DeepFace.analyze(img_path = root + row['file'],actions = ['gender', 'race', 'age'])
            hld_gend.append(tmp[0]['dominant_gender'])
            hld_race.append(tmp[0]['dominant_race'])
            hld_age.append(tmp[0]['age'])
        except ValueError:
            hld_gend.append('nan')
            hld_race.append('nan')
            hld_age.append('nan')
    
    retu_gend = util.sortit(hld_gend, ff_gend, genders)
    retu_race = util.sortit(hld_race, ff_race, races)
    retu_age = util.sortit_age(hld_age, ff_age, ages_)
    age_by_gender = util.discriminate(hld_age, ff_age, ff_gend, genders)
    age_by_race = util.discriminate(hld_age, ff_age, ff_race, races)
    
    age_diff_g = util.age_diff(hld_age, ff_age, ff_gend)
    age_diff_g.plot(kind='bar')
    age_diff_r = util.age_diff(hld_age, ff_age, ff_race)
    age_diff_r.plot(kind='bar')
    
    retu_gend.plot(kind='bar', stacked=True, title='Gender Accuracy')
    print(util.findacc(retu_gend))
    
    retu_race.plot(kind='bar', stacked=True, title='Racial Accuracy')
    print(util.findacc(retu_race))
    
    retu_age.plot(kind='bar', stacked=True, title='Age Accuracy')
    print(util.findacc(retu_age))
    
    age_by_gender.plot(kind='bar', stacked=True, title='Age by Gender Accuracy')
    print(util.findacc(age_by_gender))
    
    age_by_race.plot(kind='bar', stacked=True, title='Age by Race Accuracy')
    print(util.findacc(age_by_race))
    
    return

if __name__ == "__main__":
    main()

