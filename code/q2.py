import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data from files 
def read_data(filepath,passed_df=pd.DataFrame()):
    file=pd.read_csv(filepath,dtype="float32",low_memory=True,header=None,engine="pyarrow")
    x_points=file[0].to_numpy()
    y_points=file[1].to_numpy()
    z_points=file[2].to_numpy()
    return x_points,y_points,z_points

#finding the covariance 
def find_cov(points1,points2):
    cov= (np.sum(np.matmul((points1-np.mean(points1)),(points2-np.mean(points2))))/(points1.size-1))
    return cov
    
#LS 
def least_squares(x_points,y_points,z_points):
    
    x_matrix=np.vstack((x_points,y_points,z_points)).transpose()
    # print("x matrix shape",x_matrix.shape)
    y_matrix=np.ones(x_points.shape)
    # print("y matrix shape",y_matrix.shape)
    #Doing X.T*X^-1
    x_transpose_x=np.linalg.pinv(np.dot(x_matrix.transpose(),x_matrix))
    #Finding X.T*Y
    x_y=np.dot(x_matrix.transpose(),y_matrix)
    #Finding A 
    A=np.dot(x_transpose_x,x_y)
    return A

#TLS
def total_least_sqaures(x_points,y_points,z_points):
    #Finding X,Y and Z bar
    x_mean=np.mean(x_points)
    y_mean=np.mean(y_points)
    z_mean=np.mean(z_points)
    #Points minus the means 
    x=x_points-x_mean
    y=y_points-y_mean
    z=z_points-z_mean
    #Stacking X Y Z points , as the equation is AX+BY+CZ=1
    u=np.vstack((x,y,z)).transpose()
    # # print("shape of u TLS",u.shape)
    # matrix=np.dot(u.transpose(),u)
    # u_matrix=np.dot(matrix.transpose(),matrix)
    #Fiding U.T*U
    u_matrix=np.dot(u.transpose(),u)
    # Finding eigen values
    eigen_values,eigen_vectors=np.linalg.eig(u_matrix)
    # print("Eigen values and vectors for TLS",eigen_values,eigen_vectors)
    #min eigen value and it's vector is the solution for the coeffecient matrix
    min_eigen_vector=eigen_vectors[:,np.argmin(eigen_values)]
    return min_eigen_vector

#Finding the inlier and outlier points for each hypothesis of RANSAC
def eval_hypo(x_points,y_points,z_points,coef,threshold=0.1):
    total_post=[]
    distance=np.empty(1,)
    for i in range(x_points.shape[0]):
        #distance of hypothesis plane and points in Dataset
        dist=(np.abs((coef[0]*x_points[i])+(coef[1]*y_points[i]+(coef[2]*z_points[i])-1)))/np.sqrt(((coef[0]**2)+(coef[1]**2)+coef[2]**2))
        #Log of all distances of Points 
        distance=np.append(distance,dist)

    #Number of inliers
    success=np.where(distance<=threshold)[0].shape[0]
    # print("Pos. points",success)
    return success
#RANSAC using LS
def ransac_fitting(x_points,y_points,z_points,out_prob=0.5,success_prob=0.99,sample_points=3):
    e=out_prob
    p=success_prob
    hypo=[]
    list_of_points=np.empty(0)
    inliers=np.empty(0)
    thresh_result=[]
    # print("List of points init",list_of_points)
    samples=int(np.log(1 - p) / np.log(1 - np.power((1 - e), sample_points)))
    # print("Number of iterations",samples)
    #setting number of samples to 1000 since Calulated samples number is too low 
    samples=1000
    for i in range(0,samples):
        #getting random points 
        points=np.random.randint(0,x_points.shape[0],(3,))
        #ensuring points do not repeat
        if points not in list_of_points:
            # print(points)
            # print("Points added")
            list_of_points=np.append(list_of_points,points)
            x=x_points[points]
            y=y_points[points]
            z=z_points[points]
            # print("Points",x,y,z)
            #doing fitting
            coef=least_squares(x,y,z)
            # print("LS in RANSAC variable values",coef)
            #finding best hypo
            success=eval_hypo(x_points,y_points,z_points,coef)
            thresh_result.append(success)
            inliers=np.append(inliers,success)
            hypo.append(coef)
            # print("checking",coef,success)

        else:
            print("Points already present renewing")
    
    #Selecting the best Value 
    # print("Inlier of each hypo",inliers)
    # print("Inlier of each hypo",thresh_result)
    best_hypo=np.argmax(inliers)
    # print("List of random points taken",list_of_points)
    return hypo[best_hypo],inliers[best_hypo] 
  


# Finding the inlier and outlier points for each hypothesis of RANSAC TLS 
def eval_hypo_TLS(x_points,y_points,z_points,coef,threshold=0.1):
    total_post=[]
    distance=np.empty(1,)
    x_mean=np.mean(x_points)
    y_mean=np.mean(y_points)
    z_mean=np.mean(z_points)
    for i in range(x_points.shape[0]):
        #distance of hypothesis plane and points in Dataset
        dist=(np.abs((coef[0]*(x_points[i]-x_mean))+(coef[1]*(y_points[i]-y_mean))+(coef[2]*(z_points[i]-z_mean))))/np.sqrt(((coef[0]**2)+(coef[1]**2)+coef[2]**2))
        #Log of all distances of Points 
        distance=np.append(distance,dist)

    success=np.where(distance<=threshold)[0].shape[0]
    # print("Pos. points",success)
    return success

#RANSAC using TLS 
def ransac_fitting_TLS(x_points,y_points,z_points,out_prob=0.5,success_prob=0.99,sample_points=3):
    e=out_prob
    p=success_prob
    hypo=[]
    list_of_points=np.empty(0)
    inliers=np.empty(0)
    thresh_result=[]
    # print("List of points init",list_of_points)
    samples=int(np.log(1 - p) / np.log(1 - np.power((1 - e), sample_points)))
    print("Number of iterations",samples)
    samples=1000
    for i in range(0,samples):
        points=np.random.randint(0,x_points.shape[0],(3,))
        if points not in list_of_points:
            # print(points)
            # print("Points added")
            list_of_points=np.append(list_of_points,points)
            x=x_points[points]
            y=y_points[points]
            z=z_points[points]
            # print("Points",x,y,z)
            coef=total_least_sqaures(x,y,z)
            # print("TLS in RANSAC variable values",coef)
            success=eval_hypo(x_points,y_points,z_points,coef)
            thresh_result.append(success)
            inliers=np.append(inliers,success)
            hypo.append(coef)
            # print("checking",coef,success)

        else:
            print("Points already present renewing")
    
    # print("Inlier of each hypo",inliers)
    # print("Inlier of each hypo",thresh_result)
    best_hypo=np.argmax(inliers)
    return hypo[best_hypo],inliers[best_hypo]


file= r".\source\pc1.csv"
file1= r".\source\pc2.csv"

#reading data from frame 
x1,y1,z1=read_data(file)
x2,y2,z2=read_data(file1)


cov_matrix_1=np.array([[find_cov(x1,x1),find_cov(x1,y1),find_cov(x1,z1)],[find_cov(y1,x1),find_cov(y1,y1),find_cov(y1,z1)],[find_cov(z1,x1),find_cov(z1,y1),find_cov(z1,z1)]])
print("Covairance Matrix of PC1 \n",cov_matrix_1)

cov_matrix_2=np.array([[find_cov(x2,x2),find_cov(x2,y2),find_cov(x2,z2)],[find_cov(y2,x2),find_cov(y2,y2),find_cov(y2,z2)],[find_cov(z2,x2),find_cov(z2,y2),find_cov(z2,z2)]])
print("Covairance Matrix of PC2 \n ",cov_matrix_2)

eigen_values_1,eigen_vector_1=np.linalg.eig(cov_matrix_1)
eigen_values_2,eigen_vector_2=np.linalg.eig(cov_matrix_2)
idx1=np.argmin(eigen_values_1)
idx2=np.argmin(eigen_values_2)
print("PC1 Eigen values",eigen_values_1,"\n\nPC1 Eigen Vectors",eigen_vector_1,"\n\nPC1 The surface normal is ",eigen_vector_1[:,idx1],"\n\nPC1 Min Eigen Value",eigen_values_1[idx1])
print("PC2 Eigen values",eigen_values_2,"\n\nPC2 Eigen Vectors",eigen_vector_2,"\n\nPC2 The surface normal is ",eigen_vector_2[:,idx1],"\n\nPC2 Min Eigen Value",eigen_values_2[idx2])

a1=least_squares(x1,y1,z1)
print("least squares for PC1",a1)
p1=total_least_sqaures(x1,y1,z1)
print("total least sqaures for PC1",p1)

a2=least_squares(x2,y2,z2)
print("least squares for PC2",a2)
p2=total_least_sqaures(x2,y2,z2)
print("total least sqaures for PC2",p2)


ransac_coef,inliers=ransac_fitting(x1,y1,z1)
print("RANSAC with LS",ransac_coef,inliers)

ransac_coef_2,inliers_2=ransac_fitting(x2,y2,z2)
print("RANSAC with LS",ransac_coef_2,inliers_2)


# ax = plt.subplot2grid((6,7),(0,0),rowspan= 3,colspan= 3,projection='3d')
fig=plt.figure(1)
ax = fig.add_subplot(111,projection='3d')
xx, yy = np.meshgrid(x1,y1)
zz = (1-(a1[0]*xx)-(a1[1]*yy)) / (a1[2])
ax.plot_surface(xx, yy, zz, alpha=0.5)
ax.scatter(x1,y1,z1)
ax.set_title("Least Squares PC1")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")

# ax1 = plt.subplot2grid((6,7),(0,4),rowspan= 3,colspan= 3,projection='3d')
fig1=plt.figure(2)
ax1=fig1.add_subplot(111,projection='3d')
xx1, yy1 = np.meshgrid(x1,y1)
zz1 = np.mean(z2)-(((p1[0]*(xx1-np.mean(x1)))+(p1[1]*(yy1-np.mean(y1)))) / (p1[2]))
ax1.plot_surface(xx1, yy1, zz1, alpha=0.5)
ax1.scatter(x1,y1,z1)
ax1.set_title("Total Least Squares PC1")
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-Axis")
ax1.set_zlabel("Z-Axis")


# ax3 = plt.subplot2grid((8,7),(5,0),rowspan= 3,colspan= 3,projection='3d')
fig3=plt.figure(3)
ax3=fig3.add_subplot(111,projection='3d')
xx2, yy2 = np.meshgrid(x2,y2)
zz2 = (1-(a2[0]*xx2)-(a2[1]*yy2)) / (a2[2])
ax3.plot_surface(xx2, yy2, zz2, alpha=0.5)
ax3.scatter(x2,y2,z2)
ax3.set_title("Least Squares PC2")
ax3.set_xlabel("X-axis")
ax3.set_ylabel("Y-Axis")
ax3.set_zlabel("Z-Axis")


# ax4 = plt.subplot2grid((8,7),(5,4),rowspan= 3,colspan= 3,projection='3d')
fig4=plt.figure(4)
ax4=fig4.add_subplot(111,projection='3d')
xx2, yy2 = np.meshgrid(x2,y2)
zz2 = np.mean(z2)-(((p2[0]*(xx2-np.mean(x2)))+(p2[1]*(yy2-np.mean(y2)))) / (p2[2]))
ax4.plot_surface(xx2, yy2, zz2, alpha=0.5)
ax4.scatter(x2,y2,z2)
ax4.set_title("Total Least Squares PC2")
ax4.set_xlabel("X-axis")
ax4.set_ylabel("Y-Axis")
ax4.set_zlabel("Z-Axis")


# ax5 = plt.subplot2grid((8,7),(5,4),rowspan= 3,colspan= 3,projection='3d')
fig5=plt.figure(5)
ax5=fig5.add_subplot(111,projection='3d')
xx1, yy1 = np.meshgrid(x1,y1)
zz1 = np.mean(z1)-(((ransac_coef[0]*(xx1-np.mean(x1)))+(ransac_coef[1]*(yy1-np.mean(y1)))) / (ransac_coef[2]))
ax5.plot_surface(xx1, yy1, zz1, alpha=0.5)
ax5.scatter(x1,y1,z1)
ax5.set_title("Ransac PC1")
ax5.set_xlabel("X-axis")
ax5.set_ylabel("Y-Axis")
ax5.set_zlabel("Z-Axis")




fig6=plt.figure(6)
ax6=fig6.add_subplot(111,projection='3d')
xx2, yy2 = np.meshgrid(x2,y2)
zz2 = np.mean(z2)-(((ransac_coef_2[0]*(xx2-np.mean(x2)))+(ransac_coef_2[1]*(yy2-np.mean(y2)))) / (ransac_coef_2[2]))
ax6.plot_surface(xx2, yy2, zz2, alpha=0.5)
ax6.scatter(x2,y2,z2)
ax6.set_title("Ransac PC2")
ax6.set_xlabel("X-axis")
ax6.set_ylabel("Y-Axis")
ax6.set_zlabel("Z-Axis")


#RANSAC with TLS
ransac_coef,inliers=ransac_fitting_TLS(x1,y1,z1)
print("RANSAC with TLS",ransac_coef,inliers)

ransac_coef_2,inliers_2=ransac_fitting_TLS(x2,y2,z2)
print("RANSAC with TLS",ransac_coef_2,inliers_2)

fig7=plt.figure(7)
ax7=fig7.add_subplot(111,projection='3d')
xx1, yy1 = np.meshgrid(x1,y1)
zz1 = np.mean(z1)-(((ransac_coef[0]*(xx1-np.mean(x1)))+(ransac_coef[1]*(yy1-np.mean(y1)))) / (ransac_coef[2]))
ax7.plot_surface(xx1, yy1, zz1, alpha=0.5)
ax7.scatter(x1,y1,z1)
ax7.set_title("Ransac PC1 with TLS")
ax7.set_xlabel("X-axis")
ax7.set_ylabel("Y-Axis")
ax7.set_zlabel("Z-Axis")

fig8=plt.figure(8)
ax8=fig8.add_subplot(111,projection='3d')
xx2, yy2 = np.meshgrid(x2,y2)
zz2 = np.mean(z2)-(((ransac_coef_2[0]*(xx2-np.mean(x2)))+(ransac_coef_2[1]*(yy2-np.mean(y2)))) / (ransac_coef_2[2]))
ax8.plot_surface(xx2, yy2, zz2, alpha=0.5)
ax8.scatter(x2,y2,z2)
ax8.set_title("Ransac PC2 with TLS")
ax8.set_xlabel("X-axis")
ax8.set_ylabel("Y-Axis")
ax8.set_zlabel("Z-Axis")

plt.show()