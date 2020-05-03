# Usado para comparar o th e tw e obter o melhor Fp e Tp, a entrada consiste na imagem original 
#e o padrão ouro

'''''for th in  range (90,100):
    for tw in range(70,90):
        Image_obtida = Calvo_implementation(Image_original,tw,th)
        Image_and = np.bitwise_and(Image_obtida,Image_gold)
        Image_or = np.bitwise_or(Image_obtida,Image_gold)
        Image_or = np.bitwise_and(Image_or,Image_or)
        Image_gold_invert = np.logical_not(Image_gold_
        Image_exseg = np.logical_and(Image_gold_invert,Image_obtida)
        

        and_sum = np.sum(Image_and)
        or_sum = np.sum(Image_or)
        I_U = and_sum/or_sum
        tp = and_sum/np.sum(Image_gold)
        fp = np.sum(Image_exseg)/np.sum(Image_obtida)

        if maior[0] < I_U :
            maior[0]=I_U
            maior[1]=th
            maior[2]=tw


print(maior) #0.5580009459246413,90,88'''
'''
th_values = range(85,95,2)
tw_values = range(80,90,2)
I_U_mat = np.zeros((len(th_values),len(tw_values)))


for row,th in enumerate(th_values):
    for col,tw in enumerate(tw_values):
       print(th,tw)
       if th<tw:
        pass
       else:
        
        
        
        Image_and = np.bitwise_and(Image_obtida,Image_gold)
        Image_or = np.bitwise_or(Image_obtida,Image_gold)
        Image_or = np.bitwise_and(Image_or,Image_or)
        Image_gold_invert = np.logical_not(Image_gold)
        Image_exseg = np.logical_and(Image_gold_invert,Image_obtida)
        

        and_sum = np.sum(Image_and)
        or_sum = np.sum(Image_or)
        I_U = and_sum/or_sum
        tp = and_sum/np.sum(Image_gold)
        fp = np.sum(Image_exseg)/np.sum(Image_obtida)

        
        I_U_mat[row,col]=I_U
        


pickle.dump(I_U_mat,open("I_U_mat.dat","wb"))
#I_U_mat = pickle.load(open("I_U_mat","rb"))

plt.imshow(I_U_mat,'gray')
plt.show()
'''



