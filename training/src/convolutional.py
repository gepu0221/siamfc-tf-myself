import tensorflow as tf

#scope:指定当前所处的scope的名称，默认为None时名称即为‘conv’
def set_convolutional(X,W_shape,b_shape,stride,bn_beta,bn_gamma,bn_init_mean,bn_init_var,
                      batchnorm=True,activation=True,reuse=False,scope=None):
    
    #use the input scope or default to "conv"
    #if reuse????:?????????
    with tf.variable_scope(scope or 'conv',reuse=reuse):
    #with tf.variable_scope(scope or 'conv'):
        #trainable:标记是否加入GraphKeys.TRAINABLE_VARIABLES集合
        #tf.truncated_normal_initializer(stddev=0.1):生成的随机的标准方差*********以高斯分布的方式初始化W和b，之后复用（reuse=True)
        W=tf.get_variable("W",W_shape,trainable=False,initializer=tf.truncated_normal_initializer(stddev=0.1))
        b=tf.get_variable("b",b_shape,trainable=False,initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        #padding='VALID'：按照(图片大小-filterSize(=W.size))/stride+1
        #padding='SAME' :大小和原图像一致
        #stride:卷积的步长
        h=tf.nn.conv2d(X,W,strides=[1,stride,stride,1],padding='VALID')+b
        
        if batchnorm:
            #def tf.layers.batch_normalization(input,mean,variance,offset,scale,variance_epsilon,name=None)
            h=tf.layers.batch_normalization(h,beta_initializer=tf.constant_initializer(bn_beta),
                                           gamma_initializer=tf.constant_initializer(bn_gamma),
                                           moving_mean_initializer=tf.constant_initializer(bn_init_mean),
                                           moving_variance_initializer=tf.constant_initializer(bn_init_var),
                                           training=False,trainable=False)
        if activation:
            h=tf.nn.relu(h)
            
    return h
    