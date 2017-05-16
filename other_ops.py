'''
Created on 26 Nis 2017

@author: FIRAT
'''

#agirliklarin ciktisini almak icin...
# if i == 0 or i == epoch-1:
#         print(i)
#         hidden1 = sess.run(z1, feed_dict = {a0: [imageY,imageA,imageS,imageI,imageYY],
#                                                     y : [classY,classA,classS,classI,classY]})
#         hidden2 = sess.run(z2, feed_dict = {a0: [imageY,imageA,imageS,imageI,imageYY],
#                                                     y : [classY,classA,classS,classI,classY]})
#         hidden3 = sess.run(z3, feed_dict = {a0: [imageY,imageA,imageS,imageI,imageYY],
#                                                     y : [classY,classA,classS,classI,classY]})
#         output = sess.run(a3, feed_dict = {a0: [imageY,imageA,imageS,imageI,imageYY],
#                                                     y : [classY,classA,classS,classI,classY]})
#         nt = sess.run(a0, feed_dict = {a0: [imageY,imageA,imageS,imageI,imageYY],
#                                                     y : [classY,classA,classS,classI,classY]})
# # 
# 
#         np.savetxt("h1"+str(i)+".csv",hidden1,delimiter=",")
# 
#         np.savetxt("h2"+str(i)+".csv",hidden2,delimiter=",")
# 
#         np.savetxt("h3"+str(i)+".csv",hidden3,delimiter=",")
# 
#         np.savetxt("out"+str(i)+".csv",output,delimiter=",")
#         np.savetxt("init"+str(i)+".csv",nt,delimiter=",")

#burada test kismi yapilmaktadir.
# res = sess.run(acct_res, feed_dict =
#                         {a0: [imageY,imageA,imageS],
#                          y : [classS,classY,classS]})
# mat = sess.run(acct_mat, feed_dict =
#                         {a0: [imageY,imageA,imageS],
#                          y : [classS,classY,classS]})
#  
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
