

Theory and Implementation
==========================


Random field
--------------

Gaussian processes have a long history in the statistics community. 
They have been particularly well developed in the image signal processing community under the name random field,
and in the geostatistics community under the name of kriging 
, which is a generic name used by geostatisticians for a family of generalized least-squares regression algorithms 
in recognition of the pioneering work of a mining engineer Danie Krige :cite:`Krige:1951`
All kriging estimators are but variants of the basic linear regression estimator :cite:`Goovaerts:1997`. 
For instance, a simple kriging estimator can be defined as

.. math::
    :label: eq:estimatorSk

    Z^*({u_\alpha}) = \phi_0 + \boldsymbol{\phi}^T\boldsymbol{Z}(\boldsymbol{u})

in which :math:`\phi_0` and :math:`\boldsymbol{\phi}` are combining coefficients; 
:math:`\boldsymbol{Z}(\boldsymbol{u})` is the measured expected values at location :math:`\boldsymbol{u}`. 
If :math:`\boldsymbol{Z}(\boldsymbol{u})` is second order stationary 
(i.e., invariant and known expected value exists, noted as :math:`\boldsymbol{\mu}`), 
the observed values at location :math:`\boldsymbol{u}` could be expressed as

.. math::
    :label: eq:u_alpha

    \boldsymbol{Z}(\boldsymbol{u}) = \boldsymbol{\mu}+\boldsymbol{\epsilon}(\boldsymbol{u})


where :math:`\boldsymbol{\epsilon}(\boldsymbol{u})` is the error vector with zero mean; 
:math:`\boldsymbol{\mu}` is the invariant and known mean value of this random field. 
Hence the unobserved value at location :math:`{u}_\alpha` is

.. math::
    :label: eq:u_alpha_2

    {Z}({u}_\alpha) = {\mu}+{\epsilon}({u}_\alpha)


Simple kriging estimator should be unbiased:

.. math::
    :label: eq:simple_kriging

    E[Z^*({u_\alpha})-{Z}({u}_\alpha)] = E[\phi_0 
    + \boldsymbol{\phi}^T\boldsymbol{Z}(\boldsymbol{u})-{\mu}-{\epsilon}({u}_\alpha)]
    =\phi_0+\boldsymbol{\phi}^T\boldsymbol{u}-\mu=0


This yields 


.. math::
    :label: eq:w0

    \phi_0=\mu - \boldsymbol{\phi}^T\boldsymbol{u}



Plug :eq:`eq:w0` into :eq:`eq:estimatorSk` so that the estimator becomes

.. math::
    :label: eq:estimatorSk_new

    Z^*({u_\alpha}) = \mu+ \boldsymbol{\phi}^T  (  \boldsymbol{Z}(\boldsymbol{u})-\boldsymbol{\mu} )
    =\mu+\boldsymbol{\phi}^T\boldsymbol{\epsilon}(\boldsymbol{u})




Therefore the variance of the predicted values is 

.. math::
    :label: eq:var_Z


    \rm{Var}\{ Z^*({u_\alpha}) - Z({u_\alpha})\} 
    &=E[\mu+\boldsymbol{\phi}^T\boldsymbol{\epsilon}(\boldsymbol{u})-{\mu}-{\epsilon}({u}_\alpha)]^2  \\
    &=E[\boldsymbol{\phi}^T\boldsymbol{\epsilon}(\boldsymbol{u})-{\epsilon}({u}_\alpha)]^2  \\
    &=\rm{Var}\{\boldsymbol{\phi}^T\boldsymbol{\epsilon}(\boldsymbol{u}) \} 
    +\rm{Var}\{ {\epsilon}({u}_\alpha) \}-2\rm{COV}\{\boldsymbol{\phi}^T\boldsymbol{\epsilon}(\boldsymbol{u}),{\epsilon}({u}_\alpha) \} 
    =\boldsymbol{\phi}^T\Sigma_{\boldsymbol{u},\boldsymbol{u}}\boldsymbol{\phi} + \rm{Var}\{\boldsymbol{Z}(\boldsymbol{u})\} -2\boldsymbol{\phi}^T \Sigma_{\boldsymbol{u},\alpha} \\


To optimize :math:`\rm{Var}\{ Z^*({u_\alpha}) - Z({u_\alpha})\}`, 
let its partial derivatives with respect to the vector of coefficients :math:`\boldsymbol{\phi}` equate to zero

.. math::
    :label: eq:var_Z_partial


    \frac{\partial \left( \boldsymbol{\phi}^T\Sigma_{\boldsymbol{u},\boldsymbol{u}}\boldsymbol{\phi} 
    + \rm{Var}\{\boldsymbol{Z}(\boldsymbol{u})\} -2\boldsymbol{\phi}^T \Sigma_{\boldsymbol{u},\alpha} \right)}{\partial \boldsymbol{\phi}}
    =2\Sigma_{\boldsymbol{u},\boldsymbol{u}}\boldsymbol{\phi}-2\Sigma_{\boldsymbol{u},\alpha}=0



Hence 

.. math::
    :label: eq:phi

    \boldsymbol{\phi}=\Sigma_{\boldsymbol{u},\boldsymbol{u}}^{-1}\Sigma_{\boldsymbol{u},\alpha}


Plug \ref{eq:phi} into \ref{eq:estimatorSk_new} to obtained the simple kriging predictor

.. math::
    :label: eq:SK_estimator

    Z^*({u_\alpha}) = 
    \mu+{\Sigma_{\boldsymbol{u},\alpha}}^T\Sigma_{\boldsymbol{u},\boldsymbol{u}}^{-1}
    [\boldsymbol{Z}(\boldsymbol{u})-\boldsymbol{\mu}]


and the simple kriging variance is obtained as

.. math::
    :label: eq:SK_variance

    \rm{Var}\{ Z^*({u_\alpha}) - Z({u_\alpha})\}
    &=\boldsymbol{\phi}^T\Sigma_{\boldsymbol{u},\boldsymbol{u}}\boldsymbol{\phi} + \rm{Var}\{\boldsymbol{Z}(\boldsymbol{u})\} -2\boldsymbol{\phi}^T \Sigma_{\boldsymbol{u},\alpha} \\
    &=\boldsymbol{\phi}^T \Sigma_{\boldsymbol{u},\alpha} + \rm{Var}\{\boldsymbol{Z}(\boldsymbol{u})\} -2\boldsymbol{\phi}^T \Sigma_{\boldsymbol{u},\alpha}\\
    &=\rm{Var}\{\boldsymbol{Z}(\boldsymbol{u})\}-\Sigma_{\boldsymbol{u},\alpha}^T
    \Sigma_{\boldsymbol{u},\boldsymbol{u}}^{-1}
    \Sigma_{\boldsymbol{u},\alpha}






Neural Network
----------------


Artificial neural networks (ANNs) are a form of artificial intelligence which attempt to mimic the behavior of the human brain and nervous system.  Many researchers have described the structure and operation of ANNs 
(e.g. :cite:`Hecht:1990`; :cite:`Zurada:1992`; :cite:`Fausett:1994`). A typical structure of ANNs consists of a number of nodes (processing elements), that are usually arranged in layers: an input layer, 
an output layer and one or more hidden layers (Figure :numref:`fig:ANN`).
The input from each node in the previous layer (:math:`x_i`) is multiplied by an adjustable connection weight (:math:`w_{ji}`). At each node, the weighted input signals are summed and a threshold value (:math:`\theta_j`) is added. This combined input 
(:math:`I_j`) is then passed through a non-linear transfer function (f(.)) to produce the output of the PE (:math:`y_i`). 
The output of one PE provides the input to the nodes in the next layer. This process is summarized in :eq:`eq:pro_ann1` and :eq:`eq:pro_ann2` and illustrated in Figure :numref:`fig:ANN`.

.. math::
    :label: eq:pro_ann1

    I_j=\sum w_{ji} x_i + \theta_j


.. math::
    :label: eq:pro_ann2

    y_j = f(I_j)


.. _fig:ANN:

.. figure:: ../../images/ANN.png
	:align: center
	:figclass: align-center

	Artificial neural network structure


The ANN modelling philosophy is similar to a number of conventional statistical models in the sense that both are attempting to capture the relationship between a historical set of model inputs and corresponding outputs. ANNs learn from data examples presented to them and use these data to adjust their weights in an attempt to capture the relationship between the model input variables and the corresponding outputs. Consequently, ANNs do not need any prior knowledge about the nature of the relationship between the input/output variables, 
which is one of the benefits that ANNs have compared with most empirical and statistical methods.

ANNs have been applied in a great deal of research for a variety of purposes including medicine and biology (:cite:`Malmgren.etal:2012`; 
:cite:`Jayalakshmi.Santhakumaran:2011`); pattern recognition and image analysis (:cite:`Bishop:1995`; :cite:`Yang.etal:2000`; :cite:`Amini:2008`); 
geotechnical engineering (:cite:`Shahin.etal:2008`); 
decision making and control (:cite:`Johnson.Rogers:1998`; :cite:`Lou.Brunn:1998`; :cite:`Yanar:2007`; :cite:`Zemouri.etal:2010`); 
and stock market predictions (:cite:`Dechpichai:2010`) despite the disadvantages such as the black box nature and the 
empirical nature of model development (:cite:`Tu:1996`). 

ANNs are also widely used in spatial analysis and predictions of geotechnical and other engineering problems. 
:cite:`Sitharam.etal:2008` used ANN to evaluate the spatial variability of rock depth in an extended region. 
:cite:`Prasomphan.Mase:2013` develop a scheme to generate prediction map for geostatistical data using ANN. When integrated with GIS geographic information system (GIS), ANN is a very powerful tool to make spatial analysis. 
For example, :cite:`Lee.etal:2003` integrated  with ANN to predict the regional landslide susceptibility. More works relating to ANN-GIS can be found 
in :cite:`Gangopadhyay.etal:1999` :cite:`Yanar:2007` :cite:`vanLeeuwen.etal:2008`, etc.


Spatial predictions with geostatistic tools (such as kriging methods) usually need a prescribed spatial correlation structure, 
which should be inferred from measured data. 
This is impossible when the size of the database is small. 
One advantage of ANNs is that such a prescribed correlation structure is not needed. 
Geostatistic tools, however, are still a good and widely used method for they are able to yield relatively precise and 
spatially smooth predictions. Efforts to combine ANNs with traditional geostatistic tools have been made and can be found 
in literature :cite:`Rizzo.Dougherty:1994`, :cite:`Demyanov.etal:1998` and :cite:`Liu.etal:2009`.




.. bibliography:: references.bib

