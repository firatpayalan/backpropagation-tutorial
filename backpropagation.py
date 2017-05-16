'''
Created on 24 Nis 2017

@author: FIRAT
'''
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    img = img.convert('1')
    data = np.asarray( img, dtype="int32" ).reshape(-1)
#     print(data)
    return data
def plot(x,y,xlabel,ylabel):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

#sample larin yuklenmesi
imageY = load_image('dl_data/1.png')
imageA = load_image('dl_data/2d.png')
imageS = load_image('dl_data/3h.png')
imageI = load_image('dl_data/4a.png')
imageYY = load_image('dl_data/1a.png')

testY = load_image('dl_data/1d.png')
testA = load_image('dl_data/2e.png')
testS = load_image('dl_data/3d.png')
testI = load_image('dl_data/4d.png')
testYY = load_image('dl_data/1e.png')

#toplam 4 sinif oldugu icin her 
#bir sinif, dizinin sadece bir elemani setlenecek sekilde ayarlandi
classY=[1,0,0,0]
classA=[0,1,0,0]
classS=[0,0,1,0]
classI=[0,0,0,1]
a0 = tf.placeholder(tf.float32, [None, 900]) #input sayisi
y = tf.placeholder(tf.float32, [None, 4]) #y,a,s,i

#katman 1 noron sayisi
h1Neuron = 5 
#katman 2 noron sayisi
h2Neuron = 7
#katman 3 noron sayisi
h3Neuron = 4

#hidden layer 1
w1 = tf.Variable(tf.truncated_normal([900,h1Neuron]))
b1 = tf.Variable(tf.truncated_normal([1,h1Neuron]))

#hidden layer 2
w2 = tf.Variable(tf.truncated_normal([h1Neuron,h2Neuron]))
#hidden layer 3
w3 = tf.Variable(tf.truncated_normal([h2Neuron,h3Neuron]))
#calculated layer 1
z1 = tf.add(tf.matmul(a0, w1), b1) 
#activation function layer 1
a1 = sigma(z1) 
#calculated layer 2
z2 = tf.add(tf.matmul(a1,w2),0)
#activation function layer 2
a2 = sigma(z2)
#calculated layer 3
z3 = tf.add(tf.matmul(a2,w3),0)
#activation function layer 3
a3 = sigma(z3)

#backpropagating
diff = tf.subtract(a3,y)
#karesel hata
squarredErr = tf.reduce_mean(tf.pow(y - a3, 2))

dz3 = tf.multiply(diff,sigmaprime(z3))
dw3 = tf.matmul(tf.transpose(a2),dz3)
da2 = tf.matmul(dz3,tf.transpose(w3))

dz2 = tf.multiply(da2,sigmaprime(z2))
dw2 = tf.matmul(tf.transpose(a1),dz2)
da1 = tf.matmul(dz2,tf.transpose(w2))

dz1 = tf.multiply(da1,sigmaprime(z1))
dw1 = tf.matmul(tf.transpose(a0),dz1)
da0 = tf.matmul(dz1,tf.transpose(w1))

#ogrenme katsayisi
learningRate = tf.constant(0.5)
#noronlarin update edilmesi
step=[
        tf.assign(w1, tf.subtract(w1,tf.multiply(learningRate,dw1)))
    ,   tf.assign(w2, tf.subtract(w2,tf.multiply(learningRate,dw2)))
    ,   tf.assign(w3, tf.subtract(w3,tf.multiply(learningRate,dw3)))

]

acct_mat = tf.equal(tf.argmax(a3,1),tf.argmax(y,1)) #accurate matrix
acct_res = tf.reduce_sum(tf.cast(acct_mat,tf.float32)) #accurate resolution


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

epoch=2000
cost=[]
#for dongusu icerisinde training yapilmaktadir
for i in range(epoch):
    sess.run(step, feed_dict = {a0: [imageY,imageA,imageS,imageI,imageYY],
                                    y : [classY,classA,classS,classI,classY]})
    if i % 10 == 0:
        res = sess.run(acct_res, feed_dict = {a0: [testY,testA,testS,testI,testYY],
                                        y : [classY,classA,classS,classI,classY]})
        print(res)
    
#testing
# res = sess.run(acct_res, feed_dict =
#                             {a0: [testY,testA,testS,testI,testYY],
#                              y : [classY,classA,classS,classI,classY]})
# mat = sess.run(acct_mat, feed_dict =
#                             {a0: [testY,testA,testS,testI,testYY],
#                              y : [classY,classA,classS,classI,classY]}) 
# print(res) #resolution,correctness
# print(mat) #confusion matrix

#egitim sirasinda her iterasyonda squared error hesaplamak icin -for dongusune koyulmali-
#         cost = sess.run(squarredErr, feed_dict = {a0: [imageY,imageA,imageS],
#                                                 y : [classY,classA,classS]})
#     cost.append(sess.run(squarredErr, feed_dict = {a0: [imageY,imageA,imageS,imageI,imageYY],
#                                                      y : [classY,classA,classS,classI,classY]}))
# print(min(cost))
# print(max(cost))
# plot(list(range(epoch)),cost,'epoch','squarred err')

