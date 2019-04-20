
Linear regression (ordinary least squares) for systolic blood pressure,
using two predictor variables: gender (RIAGENDR) and age (RIDAGEYR).
Gender is treated as a quantitative variable and is coded as 1 for
males and 2 for females.

                   Generalized linear model analysis
=======================================================================
Family:   Gaussian            Link:     Identity  
Variance: Constant            Num obs:  47292     
Scale:    227.879549          
-----------------------------------------------------------------------
Variable    Parameter        SE       LCB       UCB   Z-score   P-value
-----------------------------------------------------------------------
icept        105.6458    0.6595  104.3268  106.9649  160.1817    0.0000
RIAGENDR      -3.8397    0.3674   -4.5744   -3.1049  -10.4521    0.0000
RIDAGEYR       0.4964    0.0083    0.4798    0.5131   59.7942    0.0000
-----------------------------------------------------------------------



Linear regression (ordinary least squares) for systolic blood pressure,
including ethnicity as a categorical covariate, using level 5 (other
race/multiracial) as the reference category.

                    Generalized linear model analysis
=========================================================================
Family:   Gaussian            Link:     Identity  
Variance: Constant            Num obs:  47292     
Scale:    223.349181          
-------------------------------------------------------------------------
  Variable    Parameter        SE       LCB       UCB   Z-score   P-value
-------------------------------------------------------------------------
icept          103.8917    0.7566  102.3785  105.4049  137.3131    0.0000
RIAGENDR        -3.9057    0.3638   -4.6334   -3.1781  -10.7352    0.0000
RIDAGEYR         0.4984    0.0084    0.4817    0.5152   59.5646    0.0000
RIDRETH1[3.0]    0.4462    0.5463   -0.6464    1.5387    0.8167    0.4141
RIDRETH1[1.0]    0.9441    0.6846   -0.4252    2.3134    1.3790    0.1679
RIDRETH1[4.0]    5.2583    0.5605    4.1372    6.3793    9.3810    0.0000
RIDRETH1[2.0]    0.6704    0.7138   -0.7572    2.0980    0.9392    0.3476
-------------------------------------------------------------------------



Linear regression (ordinary least squares) for systolic blood pressure,
including gender, age, ethnicity, and the interaction between gender
and age as covariates.  Ethnicity is a categorical covariate with level
5 (other race/multiracial) as the reference category.

                      Generalized linear model analysis
=============================================================================
Family:   Gaussian            Link:     Identity  
Variance: Constant            Num obs:  47292     
Scale:    221.549555          
-----------------------------------------------------------------------------
      Variable    Parameter        SE       LCB       UCB   Z-score   P-value
-----------------------------------------------------------------------------
icept              110.9802    1.2116  108.5570  113.4034   91.5979    0.0000
RIAGENDR            -8.6536    0.7315  -10.1166   -7.1905  -11.8295    0.0000
RIDAGEYR             0.3153    0.0259    0.2635    0.3671   12.1788    0.0000
RIDRETH1[3.0]        0.4482    0.5441   -0.6399    1.5364    0.8238    0.4100
RIDRETH1[1.0]        0.9754    0.6819   -0.3884    2.3392    1.4305    0.1526
RIDRETH1[4.0]        5.2657    0.5583    4.1492    6.3823    9.4323    0.0000
RIDRETH1[2.0]        0.6228    0.7109   -0.7991    2.0447    0.8760    0.3810
RIAGENDR:RIDAGEYR    0.1223    0.0164    0.0896    0.1551    7.4714    0.0000
-----------------------------------------------------------------------------



Regularized least squares regression (Lasso regression) for systolic
blood pressure, using equal penalty weights for all covariates and
zero penalty for the intercept.

        Generalized linear model analysis
==================================================
Family:   Gaussian            Link:     Identity  
Variance: Constant            Num obs:  6756      
Scale:    230.973959          
--------------------------------------------------
  Variable    Parameter
--------------------------------------------------
icept          102.1365
RIAGENDR         0.0000
RIDAGEYR         0.4211
RIDRETH1[3.0]    0.0000
RIDRETH1[1.0]    0.0000
RIDRETH1[4.0]    2.5151
RIDRETH1[2.0]    0.0000
--------------------------------------------------



Linear regression with systolic blood pressure as the outcome,
using a square root transform in the formula.

                    Generalized linear model analysis
==========================================================================
Family:   Gaussian            Link:     Identity  
Variance: Constant            Num obs:  47292     
Scale:    228.468869          
--------------------------------------------------------------------------
   Variable    Parameter        SE       LCB       UCB   Z-score   P-value
--------------------------------------------------------------------------
icept           112.1279    0.7234  110.6811  113.5748  154.9979    0.0000
RIAGENDR         -3.8932    0.3680   -4.6291   -3.1572  -10.5802    0.0000
sqrt(RIDAGEYR)    0.0057    0.0001    0.0055    0.0059   57.5952    0.0000
RIDRETH1[3.0]     0.1662    0.5533   -0.9404    1.2729    0.3004    0.7639
RIDRETH1[1.0]     0.4459    0.6920   -0.9382    1.8299    0.6443    0.5194
RIDRETH1[4.0]     5.0373    0.5671    3.9031    6.1715    8.8828    0.0000
RIDRETH1[2.0]     0.3749    0.7221   -1.0693    1.8191    0.5192    0.6037
--------------------------------------------------------------------------



Logistic regression using high blood pressure status (binary) as
the dependent variable, and gender and age as predictors.

                   Generalized linear model analysis
=======================================================================
Family:   Binomial          Link:     Logit   
Variance: Binomial          Num obs:  102718  
Scale:    1.000000          
-----------------------------------------------------------------------
Variable    Parameter        SE       LCB       UCB   Z-score   P-value
-----------------------------------------------------------------------
icept         -3.8132    0.1267   -4.0665   -3.5598  -30.0968    0.0000
RIAGENDR      -0.3762    0.0652   -0.5065   -0.2458   -5.7711    0.0000
RIDAGEYR       0.0663    0.0016    0.0631    0.0696   40.6338    0.0000
-----------------------------------------------------------------------


         Generalized linear model analysis
===================================================
Family:   Binomial          Link:     Logit   
Variance: Binomial          Num obs:  102718  
Scale:    1.000000          
---------------------------------------------------
Variable    Parameter       LCB       UCB   P-value
---------------------------------------------------
icept          0.0221    0.0171    0.0284    0.0000
RIAGENDR       0.6865    0.6026    0.7821    0.0000
RIDAGEYR       1.0686    1.0651    1.0721    0.0000
---------------------------------------------------
Parameters are shown as odds ratios



Elastic net penalized logistic regression for high blood pressure
status, with L1 and L2 penalties.  Age and gender are the predictor
variables.

      Generalized linear model analysis
==============================================
Family:   Binomial          Link:     Logit   
Variance: Binomial          Num obs:  242788  
Scale:    1.000000          
----------------------------------------------
Variable    Parameter
----------------------------------------------
icept         -0.0001
RIAGENDR       0.0000
RIDAGEYR      -0.0000
----------------------------------------------


