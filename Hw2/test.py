# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 08:44:31 2020

@author: udarici19
"""

import unittest
import reg
import pandas as pd


class regTest(unittest.TestCase):
 
    def test_reg(self):
        data = {'date':[2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000],
                'GDPPC':[59927.92983,57904.20196,56803.47243,55032.958,53106.90977,51603.49726,49883.11398,48466.82338,47099.98047,48382.55845,47975.96768,46298.73144,44114.74778,41712.80107,39496.48588,38023.16111,37133.24281,36334.90878],
                'FR':[1.7655,1.8205,1.8435,1.8625,1.8575,1.8805,1.8945,1.931,2.002,2.072,2.12,2.108,2.057,2.0515,2.0475,2.0205,2.0305,2.056],
                'FLFP':[57.03519821,56.80099869,56.67670059,56.97359848,57.20589828,57.6841011,58.10380173,58.62250137,59.19449997,59.47119904,59.30759811,59.36270142,59.25529861,59.16410065,59.50500107,59.62170029,59.78710175,59.9416008]}
        
        df = pd.DataFrame(data, columns=['date','GDPPC','FR','FLFP'])
        x = df['FLFP']
        x = df['GDPPC']
        y = df['FR']
            
    def test_betas(self):
        self.assertTrue("Cor_Coef")
        self.assertTrue(1.8435, 56.6767)
        self.assertTrue(1.8625, 56.97359848)
        self.assertTrue(1.8575, 57.20589828)
        self.assertTrue(2.072, 59.47119904)

if __name__ == '__main__':
    unittest.main()